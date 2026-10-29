"""T6 relay load harness: isolated server, live probes, and the T6 report.

T6 answers "does the relay stay healthy under streamed load?" on top of the
existing host load test. This module owns the parts that need to be more
careful than a benchmark:

* **Isolation.** Every server the harness spawns gets its own ``HOME``, data
  dir, config dir, and process log inside the run's temp dir, so no bench server
  writes into the developer's real ``~/.omnigent``.
* **Live probes.** ``GET /health`` every 250 ms and
  ``POST /v1/sessions/{id}/policies/evaluate`` every 1 s, both timed from the
  harness so their percentiles are real request latency.
* **Counters from the server's log.** The Phase 0 relay counters live in the
  server process, so the harness tails the isolated process log and parses the
  ``relay_stats {json}`` records instead of importing ``relay_stats``.
* **Baseline vs candidate.** ``--server-ref`` runs the same load against another
  checkout (git ref or directory) by pinning the subprocess ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from dev.benchmarks.omnigent.environment import BenchEnvironment
from dev.loadtest import t6_stats
from omnigent.process_logging import PROCESS_LOG_FILE_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[2]
UNKNOWN_COMMIT = "unknown"
RELAY_STATS_INTERVAL_ENV_VAR = "OMNIGENT_RELAY_STATS_INTERVAL_S"
# Sentinel Origin the server's CSRF/CSWSH guard trusts for a non-browser client.
INTERNAL_ORIGIN = "omnigent://internal"

# T6 gate from the relay brief: health stays fast and no relay failure fires.
HEALTH_P99_LIMIT_MS = 250.0
HEALTH_MAX_LIMIT_MS = 1000.0


def repo_root() -> Path:
    """Return the checkout the harness itself runs from."""
    return REPO_ROOT


# ── server source (baseline vs candidate) ──────────────────


@dataclass(frozen=True)
class ServerSource:
    """Checkout the spawned subprocesses import ``omnigent`` from."""

    path: Path
    commit: str
    created_worktree: bool


def resolve_server_ref(ref: str, *, repo_root: Path, worktree_root: Path) -> ServerSource:
    """Resolve *ref* to a checkout, creating a detached worktree for a git ref.

    :param ref: Directory path (used as-is) or git ref (``v0.20.0``, ``abc123``).
    :param repo_root: Repository whose object database holds the ref.
    :param worktree_root: Directory that receives temporary worktrees.
    :returns: The checkout and the commit it points at.
    :raises RuntimeError: If ``git worktree add`` fails.
    """
    candidate = Path(ref).expanduser()
    if candidate.is_dir():
        return ServerSource(
            path=candidate.resolve(), commit=git_head(candidate), created_worktree=False
        )
    worktree_root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", ref).strip("-")[:40] or "ref"
    worktree = worktree_root / f"server-{slug}-{uuid.uuid4().hex[:6]}"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree), ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git worktree add failed for {ref!r}: {detail}")
    return ServerSource(path=worktree.resolve(), commit=git_head(worktree), created_worktree=True)


def release_server_source(source: ServerSource, *, repo_root: Path) -> None:
    """Remove a worktree created by :func:`resolve_server_ref`; leave directories."""
    if not source.created_worktree:
        return
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(source.path)],
        capture_output=True,
        text=True,
        check=False,
    )
    shutil.rmtree(source.path, ignore_errors=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "prune"],
        capture_output=True,
        text=True,
        check=False,
    )


def git_head(path: Path) -> str:
    """Return the short commit at *path*, or :data:`UNKNOWN_COMMIT`."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return UNKNOWN_COMMIT
    return result.stdout.strip() or UNKNOWN_COMMIT


# ── isolated server environment ────────────────────────────


def build_server_env(
    base_env: Mapping[str, str],
    *,
    run_root: Path,
    relay_stats_interval_s: float,
    server_ref: Path | None = None,
) -> dict[str, str]:
    """Return the environment for the server under test.

    The server gets its own ``HOME`` and Omnigent data/config dirs under
    *run_root*, plus an explicit process log there. ``apply_server_env`` (which
    the benchmark calls first) strips the caller's log-file variable, so set it
    again here, for the server process only.

    :param base_env: Benchmark environment, already filtered of caller vars.
    :param run_root: Run temp dir that holds the isolated server state.
    :param relay_stats_interval_s: Cadence of ``relay_stats`` log records.
    :param server_ref: Checkout to import ``omnigent`` from, when comparing refs.
    :returns: A new environment dict; *base_env* is not modified.
    """
    server_home = run_root / "server-home"
    env = dict(base_env)
    env["HOME"] = str(server_home)
    env[PROCESS_LOG_FILE_ENV_VAR] = str(run_root / "server-process.log")
    env["OMNIGENT_DATA_DIR"] = str(server_home / "data")
    env["OMNIGENT_CONFIG_HOME"] = str(server_home / "config")
    env[RELAY_STATS_INTERVAL_ENV_VAR] = f"{float(relay_stats_interval_s):g}"
    env["OMNIGENT_NO_UPDATE_CHECK"] = "1"
    if server_ref is not None:
        env = pin_pythonpath(env, server_ref)
    return env


def pin_pythonpath(env: Mapping[str, str], source_root: Path) -> dict[str, str]:
    """Return *env* whose ``PYTHONPATH`` imports ``omnigent`` from *source_root*.

    ``BenchEnvironment`` prepends this checkout to ``PYTHONPATH`` so servers
    import branch source. Replace that prepend with *source_root* so a
    ``--server-ref`` run really executes the other checkout.
    """
    target = _normalized_path(source_root)
    own = _normalized_path(REPO_ROOT)
    parts = [
        part
        for part in env.get("PYTHONPATH", "").split(os.pathsep)
        if part and _normalized_path(part) not in {target, own}
    ]
    pinned = dict(env)
    pinned["PYTHONPATH"] = os.pathsep.join([target, *parts])
    return pinned


def _normalized_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


class T6BenchEnvironment(BenchEnvironment):
    """Bench environment whose spawned server logs into the run's temp dir."""

    def __init__(
        self,
        *,
        relay_stats_interval_s: float = 5.0,
        server_source: ServerSource | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.relay_stats_interval_s = relay_stats_interval_s
        self.server_source = server_source
        # ``_tmp`` is deleted on teardown, so artifacts must be copied out
        # before the context exits.
        self.run_root = self._tmp / "t6"
        self.server_process_log = self.run_root / "server-process.log"

    @property
    def server_pid(self) -> int:
        """PID of the running server subprocess."""
        proc = self._server_proc
        if proc is None:
            raise RuntimeError("server is not running")
        return proc.pid

    def _spawn_server(
        self,
        port: int,
        base_env: dict[str, str],
        binding_token: str,
        artifact_dir: Path,
    ) -> subprocess.Popen[bytes]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "server-home").mkdir(parents=True, exist_ok=True)
        server_env = build_server_env(
            base_env,
            run_root=self.run_root,
            relay_stats_interval_s=self.relay_stats_interval_s,
            server_ref=self.server_source.path if self.server_source is not None else None,
        )
        return super()._spawn_server(port, server_env, binding_token, artifact_dir)


# ── live probes and server monitoring ──────────────────────


@dataclass(frozen=True)
class ProbeConfig:
    """Cadence and timeouts for the T6 probes."""

    health_interval_s: float = 0.25
    policy_interval_s: float = 1.0
    probe_timeout_s: float = 10.0
    log_poll_interval_s: float = 0.5


_POLICY_EVENT = {
    "type": "PHASE_TOOL_CALL",
    "target": "",
    "data": {"name": "Bash", "arguments": {"command": "true"}},
    "context": {},
}


class T6Monitor:
    """Probe the server while load runs and collect its health signals."""

    def __init__(
        self,
        *,
        base_url: str,
        policy_session_id: str,
        server_log_path: Path,
        config: ProbeConfig,
    ) -> None:
        self._base_url = base_url
        self._policy_session_id = policy_session_id
        self._config = config
        self._tail = t6_stats.ServerLogTail(server_log_path)
        self._client: httpx.AsyncClient | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.health_samples: list[t6_stats.LatencySample] = []
        self.policy_samples: list[t6_stats.LatencySample] = []
        self.log_counters = t6_stats.LogCounters()
        self.relay = t6_stats.RelayStatsCollector()
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    @property
    def policy_session_id(self) -> str:
        """Session the policy-evaluate probe targets."""
        return self._policy_session_id

    @property
    def elapsed_seconds(self) -> float:
        """Wall time between :meth:`start` and :meth:`stop`."""
        if self._started_at is None:
            return 0.0
        end = self._stopped_at if self._stopped_at is not None else time.monotonic()
        return round(end - self._started_at, 3)

    async def start(self) -> None:
        """Start the probe and log-watch tasks."""
        self._started_at = time.monotonic()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._config.probe_timeout_s,
            headers={"Origin": INTERNAL_ORIGIN},
        )
        self._tasks = [
            asyncio.create_task(self._health_loop(), name="t6-health-probe"),
            asyncio.create_task(self._policy_loop(), name="t6-policy-probe"),
            asyncio.create_task(self._log_loop(), name="t6-log-watch"),
        ]

    async def stop(self, *, drain_seconds: float = 0.0) -> None:
        """Stop probing, optionally wait for a final log record, then close."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if drain_seconds > 0:
            await asyncio.sleep(drain_seconds)
        self.poll_server_log()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._stopped_at = time.monotonic()

    def poll_server_log(self) -> None:
        """Fold the lines appended since the last poll into the counters."""
        for line in self._tail.poll():
            self.log_counters.observe(line)
            payload = t6_stats.parse_relay_stats_line(line)
            if payload is not None:
                self.relay.observe(payload)

    def report_sections(self) -> dict[str, Any]:
        """Return the probe, event-loop, and relay-counter report sections."""
        return {
            "health": t6_stats.summarize_latency(self.health_samples).as_dict(),
            "policy_evaluate": t6_stats.summarize_latency(self.policy_samples).as_dict(),
            "event_loop": {
                "stall_reports": self.log_counters.watchdog_stalls,
                "max_stalled_s": round(self.log_counters.max_stalled_s, 3),
                "source": "server log (event_loop_watchdog)",
            },
            "relay_counters": self.relay.summary(
                window_seconds=self.elapsed_seconds,
                log_counters=self.log_counters,
            ),
        }

    async def _health_loop(self) -> None:
        while True:
            started = time.perf_counter()
            assert self._client is not None
            try:
                response = await self._client.get("/health")
                sample = t6_stats.LatencySample(_elapsed_ms(started), response.status_code)
            except httpx.HTTPError as exc:
                sample = t6_stats.LatencySample(
                    _elapsed_ms(started), None, f"{type(exc).__name__}: {exc}"
                )
            self.health_samples.append(sample)
            await asyncio.sleep(self._config.health_interval_s)

    async def _policy_loop(self) -> None:
        path = f"/v1/sessions/{self._policy_session_id}/policies/evaluate"
        while True:
            started = time.perf_counter()
            assert self._client is not None
            try:
                response = await self._client.post(path, json={"event": _POLICY_EVENT})
                sample = t6_stats.LatencySample(_elapsed_ms(started), response.status_code)
            except httpx.HTTPError as exc:
                sample = t6_stats.LatencySample(
                    _elapsed_ms(started), None, f"{type(exc).__name__}: {exc}"
                )
            self.policy_samples.append(sample)
            await asyncio.sleep(self._config.policy_interval_s)

    async def _log_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.log_poll_interval_s)
            self.poll_server_log()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


class ServerCpuSampler:
    """Sample the server process's CPU use while load runs."""

    def __init__(self, pid: int, *, interval_s: float = 1.0) -> None:
        self._pid = pid
        self._interval_s = interval_s
        self._proc: psutil.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._cpu_pct: list[float] = []
        self._rss_bytes: list[int] = []
        self._start_monotonic: float | None = None
        self._end_monotonic: float | None = None
        self._start_times: tuple[float, float] | None = None
        self._end_times: tuple[float, float] | None = None
        self.error: str | None = None

    async def start(self) -> None:
        """Attach to the server process and start sampling."""
        try:
            self._proc = psutil.Process(self._pid)
            self._proc.cpu_percent(None)  # baseline; the first call returns 0.0
            self._start_times = self._read_times()
        except psutil.Error as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._proc = None
        self._start_monotonic = time.monotonic()
        self._task = asyncio.create_task(self._sample_loop(), name="t6-cpu-sampler")

    async def stop(self) -> None:
        """Stop sampling and read the final CPU totals."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._end_monotonic = time.monotonic()
        self._end_times = self._read_times()

    def summary(self, *, events_total: int | None) -> dict[str, Any]:
        """Return the server-CPU report section.

        ``cpu_seconds_per_1k_events`` is measured CPU time per 1,000 relay
        events/s of offered load, which is the comparison figure across load
        levels and refs.
        """
        cpu_seconds: float | None = None
        if self._start_times is not None and self._end_times is not None:
            cpu_seconds = round(
                (self._end_times[0] - self._start_times[0])
                + (self._end_times[1] - self._start_times[1]),
                3,
            )
        wall_seconds = (
            round(self._end_monotonic - self._start_monotonic, 3)
            if self._start_monotonic is not None and self._end_monotonic is not None
            else 0.0
        )
        return {
            "pid": self._pid,
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "mean_cpu_pct": round(sum(self._cpu_pct) / len(self._cpu_pct), 1)
            if self._cpu_pct
            else None,
            "max_cpu_pct": round(max(self._cpu_pct), 1) if self._cpu_pct else None,
            "rss_max_bytes": max(self._rss_bytes) if self._rss_bytes else None,
            "cpu_samples": len(self._cpu_pct),
            "events_total": events_total,
            "cpu_seconds_per_1k_events": t6_stats.cpu_seconds_per_1k_events(
                cpu_seconds, events_total
            ),
            "error": self.error,
        }

    def _read_times(self) -> tuple[float, float] | None:
        if self._proc is None:
            return None
        try:
            times = self._proc.cpu_times()
        except psutil.Error:
            return None
        return (times.user, times.system)

    async def _sample_loop(self) -> None:
        assert self._proc is not None
        while True:
            try:
                self._cpu_pct.append(self._proc.cpu_percent(None))
                self._rss_bytes.append(self._proc.memory_info().rss)
            except psutil.Error as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            await asyncio.sleep(self._interval_s)


# ── report ─────────────────────────────────────────────────


def build_report(
    *,
    config: Mapping[str, Any],
    server: Mapping[str, Any],
    locust_exit_code: int,
    health: t6_stats.LatencySummary | Mapping[str, Any],
    policy: t6_stats.LatencySummary | Mapping[str, Any],
    event_loop: Mapping[str, Any],
    relay_counters: Mapping[str, Any],
    server_cpu: Mapping[str, Any],
    policy_session_id: str,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the T6 JSON report."""
    report: dict[str, Any] = {
        "scenario": "t6",
        "generated_at": datetime.now(UTC).isoformat(),
        "server": dict(server),
        "config": dict(config),
        "policy_probe_session_id": policy_session_id,
        "load": {
            "locust_exit_code": locust_exit_code,
            "completed": locust_exit_code == 0,
        },
        "health": health.as_dict()
        if isinstance(health, t6_stats.LatencySummary)
        else dict(health),
        "policy_evaluate": (
            policy.as_dict() if isinstance(policy, t6_stats.LatencySummary) else dict(policy)
        ),
        "event_loop": dict(event_loop),
        "relay_counters": dict(relay_counters),
        "server_cpu": dict(server_cpu),
        "artifacts": dict(artifacts),
    }
    report["gate"] = evaluate_gate(report)
    return report


def evaluate_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the T6 gate against the report's measured numbers.

    A check reads ``unknown`` when the run could not measure the signal, for
    example a ref whose ``relay_stats`` payload has no fuse counter yet. Unknown
    is not a pass: the gate only passes when every check measured a good value.
    """
    health = _mapping(report.get("health"))
    event_loop = _mapping(report.get("event_loop"))
    relay = _mapping(report.get("relay_counters"))
    checks = {
        "health_p99_ms": _latency_check(health.get("p99_ms"), HEALTH_P99_LIMIT_MS),
        "health_max_lt_1s": _latency_check(health.get("max_ms"), HEALTH_MAX_LIMIT_MS),
        "no_watchdog_stalls": _zero_check(event_loop.get("stall_reports"), "server log"),
        "no_fuse_trips": _zero_check(relay.get("fuse_trips"), relay.get("fuse_trips_source")),
        "no_persist_failures": _zero_check(
            relay.get("relay_persist_failures"), relay.get("relay_persist_failures_source")
        ),
        "no_subscriber_overflows": _zero_check(
            relay.get("subscriber_overflows"), relay.get("subscriber_overflows_source")
        ),
    }
    return {
        "thresholds": {"health_p99_ms": HEALTH_P99_LIMIT_MS, "health_max_ms": HEALTH_MAX_LIMIT_MS},
        "checks": checks,
        "pass": all(check["status"] == "pass" for check in checks.values()),
    }


def write_t6_report(out_dir: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write ``t6_report.json`` and ``t6_report.md`` into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "t6_report.json"
    markdown_path = out_dir / "t6_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))
    return json_path, markdown_path


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the human-readable T6 report."""
    server = _mapping(report.get("server"))
    config = _mapping(report.get("config"))
    health = _mapping(report.get("health"))
    policy = _mapping(report.get("policy_evaluate"))
    event_loop = _mapping(report.get("event_loop"))
    relay = _mapping(report.get("relay_counters"))
    body_queue = _mapping(relay.get("body_queue"))
    cpu = _mapping(report.get("server_cpu"))
    gate = _mapping(report.get("gate"))

    lines: list[str] = [
        "# T6 relay load report",
        "",
        f"- Generated: {report.get('generated_at', '-')}",
        f"- Server: commit `{server.get('commit', '-')}`"
        + (f" (ref `{server['ref']}`)" if server.get("ref") else "")
        + f" at {server.get('base_url', '-')}",
        f"- Load: {config.get('users', '-')} users × {config.get('sessions_per_user', '-')} "
        f"sessions × {config.get('turns_per_session', '-')} turns, "
        f"~{config.get('reply_words', '-')}-word streamed replies, "
        f"{config.get('run_time', '-')}",
        f"- Locust exit: {_mapping(report.get('load')).get('locust_exit_code', '-')}",
        "",
        "## Probe latency",
        "",
        "| Probe | Samples | OK | non-200 | p50 ms | p99 ms | max ms | mean ms |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    lines.append(_latency_row("GET /health (250 ms cadence)", health))
    lines.append(_latency_row("POST .../policies/evaluate (1 s cadence)", policy))
    lines.extend(
        [
            "",
            "## Server health signals",
            "",
            "| Signal | Value | Source |",
            "|---|---:|---|",
            f"| Event-loop watchdog stalls | {event_loop.get('stall_reports', 0)} | "
            f"{event_loop.get('source', 'server log')} |",
            f"| Max stalled seconds | {event_loop.get('max_stalled_s', 0.0)} | server log |",
            f"| Body-queue fuse trips | {_counter(relay, 'fuse_trips')} | "
            f"{relay.get('fuse_trips_source', 'unreported')} |",
            f"| Relay persistence failures | {_counter(relay, 'relay_persist_failures')} | "
            f"{relay.get('relay_persist_failures_source', 'unreported')} |",
            f"| Stream subscriber overflow closes | {_counter(relay, 'subscriber_overflows')} | "
            f"{relay.get('subscriber_overflows_source', 'unreported')} |",
            "",
            "## Relay event source mix",
            "",
            "| Event source | Events | Share % | Events/s |",
            "|---|--:|--:|--:|",
        ]
    )
    mix = _mapping(relay.get("event_source_mix"))
    share = _mapping(relay.get("event_source_share_pct"))
    per_second = _mapping(relay.get("event_source_per_second"))
    for source in sorted(mix):
        cells = (source, mix[source], share.get(source, "-"), per_second.get(source, "-"))
        lines.append("| {} | {} | {} | {} |".format(*cells))
    if not mix:
        lines.append("| (no relay_stats samples) | - | - | - |")
    lines.extend(
        [
            "",
            f"- Events total: {relay.get('events_total', 0)} "
            f"({relay.get('events_per_second', '-')}/s over "
            f"{config.get('measured_seconds', '-')}s)",
            f"- relay_stats samples: {relay.get('samples', 0)}",
            f"- Body queue peaks: depth {body_queue.get('max_depth', '-')}, "
            f"bytes {body_queue.get('max_bytes', '-')}",
            f"- Body queue guardrails: {body_queue.get('warned_requests', '-')} warned "
            f"request(s), fuse at {body_queue.get('fuse_threshold_bytes', '-')} bytes",
            f"- Outbound queue peak: "
            f"{_mapping(relay.get('outbound_queue')).get('max_depth', '-')} frame(s) across "
            f"{_mapping(relay.get('outbound_queue')).get('runners_tracked', '-')} runner(s)",
            f"- Subscriber serializations: "
            f"{_mapping(relay.get('subscriber_serialization')).get('count', '-')} "
            f"(max {_mapping(relay.get('subscriber_serialization')).get('seconds_max', '-')} s)",
            f"- Relay stretches: {_mapping(relay.get('relay_stretch')).get('count', '-')} "
            f"(max {_mapping(relay.get('relay_stretch')).get('seconds_max', '-')} s)",
            "",
            "## Server CPU",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| CPU seconds | {cpu.get('cpu_seconds', '-')} |",
            f"| Wall seconds | {cpu.get('wall_seconds', '-')} |",
            f"| Mean CPU % | {cpu.get('mean_cpu_pct', '-')} |",
            f"| Max CPU % | {cpu.get('max_cpu_pct', '-')} |",
            f"| Max RSS bytes | {cpu.get('rss_max_bytes', '-')} |",
            f"| Relay events | {cpu.get('events_total', '-')} |",
            f"| CPU / 1,000 events (offered events/s) | "
            f"{cpu.get('cpu_seconds_per_1k_events', '-')} |",
            "",
            "## T6 gate",
            "",
            "| Check | Status | Observed | Required |",
            "|---|---|---:|---:|",
        ]
    )
    for name, check in _mapping(gate.get("checks")).items():
        lines.append(
            f"| {name} | {check.get('status', '-')} | {check.get('observed', '-')} | "
            f"{check.get('required', '-')} |"
        )
    outcome = "PASS" if gate.get("pass") else "FAIL"
    lines.extend(["", f"**T6 gate: {outcome}.**", ""])
    artifacts = _mapping(report.get("artifacts"))
    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for name, value in artifacts.items():
            lines.append(f"- {name}: `{value}`")
        lines.append("")
    return "\n".join(lines)


def _latency_row(label: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {summary.get('count', 0)} | {summary.get('ok', 0)} | "
        f"{summary.get('non_200', 0)} | {_dash(summary.get('p50_ms'))} | "
        f"{_dash(summary.get('p99_ms'))} | {_dash(summary.get('max_ms'))} | "
        f"{_dash(summary.get('mean_ms'))} |"
    )


def _latency_check(value: object, limit: float) -> dict[str, Any]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return {"status": "unknown", "observed": None, "required": f"< {limit}"}
    return {
        "status": "pass" if value < limit else "fail",
        "observed": value,
        "required": f"< {limit}",
    }


def _zero_check(value: object, source: object) -> dict[str, Any]:
    if isinstance(source, str) and source == "unreported":
        return {"status": "unknown", "observed": None, "required": "0 (counter not reported)"}
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return {"status": "unknown", "observed": None, "required": "0"}
    return {"status": "pass" if value == 0 else "fail", "observed": value, "required": "0"}


def _counter(relay: Mapping[str, Any], key: str) -> object:
    value = relay.get(key)
    return "-" if value is None else value


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def copy_server_log(source: Path, out_dir: Path) -> Path | None:
    """Copy the isolated server log beside the report before teardown deletes it."""
    if not source.is_file():
        return None
    target = out_dir / "server-process.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        shutil.copyfile(source, target)
        return target
    return None
