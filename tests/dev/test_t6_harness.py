"""Tests for the T6 relay load harness helpers.

The harness itself needs a live server, so its unit tests cover the pure
pieces: parsing the server's ``relay_stats`` records, latency aggregation,
CPU-per-event normalisation, the isolated server environment, and the report
writer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dev.loadtest import t6, t6_stats
from omnigent.server import relay_stats

# ── fixtures ────────────────────────────────────────────────


def _relay_stats_payload(
    *,
    relay_sse: int = 15,
    event_batch: int = 4,
    http_post: int = 1,
    persist_failures: int = 0,
    max_depth: int = 3,
    max_bytes: int = 587,
    subscriber_serializations: int = 19,
    relay_stretches: int = 15,
) -> dict[str, object]:
    """Build a ``relay_stats`` payload shaped like the server's log record."""
    return {
        "window_seconds": 2.0,
        "body_queue": {
            "depth": 0,
            "bytes": 0,
            "max_depth": max_depth,
            "max_bytes": max_bytes,
            "requests_tracked": 2,
            "dropped_requests": 0,
            "requests": {"req-1": {"depth": 2, "bytes": 300}},
        },
        "events": {
            "by_source": {
                "relay_sse": relay_sse,
                "event_batch": event_batch,
                "http_post": http_post,
            },
            "by_source_per_second": {
                "relay_sse": relay_sse / 2.0,
                "event_batch": event_batch / 2.0,
                "http_post": http_post / 2.0,
            },
            "by_type": {
                "response.output_text.delta": {"count": relay_sse, "per_second": 7.5},
                "session.status": {"count": event_batch, "per_second": 2.0},
            },
            "sessions": {
                "conv_a": {"total": relay_sse + event_batch, "per_second": 1.0, "types": {}},
            },
            "dropped_sessions": 0,
            "dropped_event_types": 0,
        },
        "subscriber_serialization": {
            "count": subscriber_serializations,
            "seconds_total": 0.125,
            "seconds_max": 0.02,
            "sessions": {},
        },
        "relay_stretch": {
            "count": relay_stretches,
            "seconds_total": 0.3,
            "seconds_max": 0.05,
            "sessions": {},
        },
        "relay_persist_failures": {"total": persist_failures, "sessions": {}},
    }


def _log_line(payload: dict[str, object]) -> str:
    """Render *payload* the way ``configure_process_logging`` writes it."""
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return (
        "INFO 10-28 12:00:00.000 omnigent.server.relay_stats "
        f"log_relay_stats_snapshot | relay_stats {body}"
    )


# ── relay_stats parsing ─────────────────────────────────────


def test_parse_relay_stats_line_returns_payload() -> None:
    payload = _relay_stats_payload()

    parsed = t6_stats.parse_relay_stats_line(_log_line(payload))

    assert parsed == payload


def test_parse_relay_stats_line_ignores_other_records() -> None:
    assert t6_stats.parse_relay_stats_line("INFO something else") is None
    assert t6_stats.parse_relay_stats_line("relay_stats {not json") is None
    assert t6_stats.parse_relay_stats_line("relay_stats [1, 2]") is None


def test_collector_summarises_event_mix_and_counters() -> None:
    collector = t6_stats.RelayStatsCollector()
    collector.observe(_relay_stats_payload(relay_sse=30, event_batch=8, http_post=2))

    summary = collector.summary(window_seconds=4.0)

    assert summary["available"] is True
    assert summary["samples"] == 1
    assert summary["events_total"] == 40
    assert summary["event_source_mix"] == {"relay_sse": 30, "event_batch": 8, "http_post": 2}
    assert summary["event_source_share_pct"]["relay_sse"] == 75.0
    assert summary["event_source_per_second"]["relay_sse"] == 7.5
    assert summary["events_per_second"] == 10.0
    assert summary["event_type_mix"]["response.output_text.delta"] == 30
    assert summary["body_queue"]["max_depth"] == 3
    assert summary["body_queue"]["max_bytes"] == 587
    assert summary["subscriber_serialization"]["count"] == 19
    assert summary["relay_stretch"]["count"] == 15
    assert summary["relay_persist_failures"] == 0
    assert summary["relay_persist_failures_source"] == "counter"


def test_collector_reports_unavailable_without_samples() -> None:
    summary = t6_stats.RelayStatsCollector().summary(window_seconds=1.0)

    assert summary["available"] is False
    assert summary["samples"] == 0
    assert summary["events_total"] == 0
    assert summary["event_source_mix"] == {}
    assert summary["events_per_second"] is None
    assert summary["relay_persist_failures_source"] == "unreported"


def test_collector_reads_phase_two_counters_when_present() -> None:
    payload = _relay_stats_payload(persist_failures=2)
    payload["body_queue"]["fuse"] = {"trips": 1, "threshold_bytes": 268435456}
    payload["session_stream"] = {"subscriber_overflows": 3, "max_events": 1024}
    collector = t6_stats.RelayStatsCollector()

    collector.observe(payload)
    summary = collector.summary(window_seconds=2.0)

    assert summary["fuse_trips"] == 1
    assert summary["fuse_trips_source"] == "counter"
    assert summary["subscriber_overflows"] == 3
    assert summary["subscriber_overflows_source"] == "counter"
    assert summary["relay_persist_failures"] == 2


def test_collector_falls_back_to_log_counts_when_counters_absent() -> None:
    log_counters = t6_stats.LogCounters(
        fuse_log_trips=1,
        subscriber_overflow_logs=2,
    )
    collector = t6_stats.RelayStatsCollector()
    collector.observe(_relay_stats_payload())

    summary = collector.summary(window_seconds=2.0, log_counters=log_counters)

    assert summary["fuse_trips"] == 1
    assert summary["fuse_trips_source"] == "log"
    assert summary["subscriber_overflows"] == 2
    assert summary["subscriber_overflows_source"] == "log"


def test_collector_reports_unreported_counters_as_zero() -> None:
    collector = t6_stats.RelayStatsCollector()
    collector.observe(_relay_stats_payload())

    summary = collector.summary(window_seconds=2.0)

    assert summary["fuse_trips"] == 0
    assert summary["fuse_trips_source"] == "unreported"
    assert summary["subscriber_overflows"] == 0
    assert summary["subscriber_overflows_source"] == "unreported"


def test_server_payload_reports_guardrail_counters_without_log_lines() -> None:
    """The real recorder payload alone supplies the fuse and overflow counters."""
    stats = relay_stats.RelayStats(warn_threshold_bytes=10, fuse_threshold_bytes=10)
    stats.record_body_queue_enqueue("req-guarded", 32)
    stats.record_outbound_queue_depth("runner-1", 2)
    stats.record_subscriber_overflow()
    payload = relay_stats.build_relay_stats_log_payload(stats.snapshot())
    parsed = t6_stats.parse_relay_stats_line(_log_line(payload))
    assert parsed == payload

    collector = t6_stats.RelayStatsCollector()
    collector.observe(parsed)
    summary = collector.summary(
        window_seconds=2.0,
        log_counters=t6_stats.LogCounters(fuse_log_trips=9, subscriber_overflow_logs=9),
    )

    assert summary["fuse_trips"] == 1
    assert summary["fuse_trips_source"] == "counter"
    assert summary["subscriber_overflows"] == 1
    assert summary["subscriber_overflows_source"] == "counter"
    assert summary["outbound_queue"]["max_depth"] == 2
    assert summary["body_queue"]["fuse_threshold_bytes"] == 10


# ── server log observation ─────────────────────────────────


def test_log_counters_count_watchdog_fuse_and_overflow_records() -> None:
    counters = t6_stats.LogCounters()

    counters.observe(
        "WARN 10-28 12:00:00.000 omnigent.util.event_loop_watchdog watch | "
        "Server event loop has not advanced for 6.250s"
    )
    counters.observe(
        "WARN 10-28 12:00:01.000 omnigent.util.event_loop_watchdog watch | "
        "Runner event loop has not advanced for 7.000s"
    )
    counters.observe(
        "WARN 10-28 12:00:02.000 omnigent.server.routes._sessions.helpers "
        "stream | session stream subscriber overflowed for conv_a; closing"
    )
    counters.observe(
        "ERROR 10-28 12:00:03.000 omnigent.runner.transports.ws_tunnel "
        "registry | body-queue fuse tripped for request req-1"
    )
    counters.observe(
        "INFO 10-28 12:00:04.000 omnigent.server.relay_stats "
        f"log_relay_stats_snapshot | relay_stats {json.dumps({'a': 1})}"
    )

    assert counters.watchdog_stalls == 1
    assert counters.max_stalled_s == pytest.approx(6.25)
    assert counters.subscriber_overflow_logs == 1
    assert counters.fuse_log_trips == 1


def test_server_log_tail_reads_each_complete_line_once(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("first line\nsecond li")
    tail = t6_stats.ServerLogTail(log_path)

    assert tail.poll() == ["first line"]

    with log_path.open("a") as handle:
        handle.write("ne\nthird line\n")
    assert tail.poll() == ["second line", "third line"]
    assert tail.poll() == []


def test_server_log_tail_handles_a_missing_file(tmp_path: Path) -> None:
    tail = t6_stats.ServerLogTail(tmp_path / "absent.log")

    assert tail.poll() == []


# ── latency and CPU aggregation ────────────────────────────


def test_latency_summary_reports_percentiles_and_non_200_count() -> None:
    samples = [
        t6_stats.LatencySample(latency_ms=float(value), status_code=200)
        for value in (1, 2, 3, 4, 100)
    ]
    samples.append(t6_stats.LatencySample(latency_ms=250.0, status_code=503))
    samples.append(
        t6_stats.LatencySample(latency_ms=5000.0, status_code=None, error="ReadTimeout")
    )

    summary = t6_stats.summarize_latency(samples)

    assert summary.count == 7
    assert summary.ok == 5
    assert summary.non_200 == 2
    assert summary.errors == 1
    assert summary.p50_ms == 4.0
    assert summary.p99_ms == 5000.0
    assert summary.max_ms == 5000.0
    assert summary.as_dict()["non_200"] == 2


def test_latency_summary_is_empty_without_samples() -> None:
    summary = t6_stats.summarize_latency([])

    assert summary.count == 0
    assert summary.p50_ms is None
    assert summary.p99_ms is None
    assert summary.max_ms is None


def test_cpu_seconds_per_1k_events_normalises_and_guards_zero() -> None:
    assert t6_stats.cpu_seconds_per_1k_events(2.0, 5000) == pytest.approx(0.4)
    assert t6_stats.cpu_seconds_per_1k_events(2.0, 0) is None
    assert t6_stats.cpu_seconds_per_1k_events(2.0, None) is None


# ── isolated server environment ────────────────────────────


def test_build_server_env_isolates_log_home_and_counters(tmp_path: Path) -> None:
    base = {"PATH": "/usr/bin", "HOME": str(tmp_path / "caller-home")}
    run_root = tmp_path / "run"

    env = t6.build_server_env(
        base,
        run_root=run_root,
        relay_stats_interval_s=2.0,
    )

    assert base == {"PATH": "/usr/bin", "HOME": str(tmp_path / "caller-home")}
    assert env["HOME"] == str(run_root / "server-home")
    assert env[t6.PROCESS_LOG_FILE_ENV_VAR] == str(run_root / "server-process.log")
    assert env["OMNIGENT_DATA_DIR"].startswith(str(run_root))
    assert env["OMNIGENT_CONFIG_HOME"].startswith(str(run_root))
    assert env["OMNIGENT_RELAY_STATS_INTERVAL_S"] == "2"


def test_build_server_env_pins_pythonpath_to_server_ref(tmp_path: Path) -> None:
    source = tmp_path / "worktree"
    source.mkdir()
    base = {"PYTHONPATH": f"{t6.repo_root()}{os.pathsep}/somewhere"}

    env = t6.build_server_env(
        base,
        run_root=tmp_path / "run",
        relay_stats_interval_s=5.0,
        server_ref=source,
    )

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(source)
    assert str(t6.repo_root()) not in parts
    assert "/somewhere" in parts


def test_resolve_server_ref_accepts_an_existing_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    source = t6.resolve_server_ref(
        str(checkout), repo_root=tmp_path, worktree_root=tmp_path / "wt"
    )

    assert source.path == checkout.resolve()
    assert source.created_worktree is False
    assert source.commit == t6.UNKNOWN_COMMIT


def test_release_server_source_leaves_real_checkouts_alone(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = t6.ServerSource(path=checkout, commit="deadbee", created_worktree=False)

    t6.release_server_source(source, repo_root=tmp_path)

    assert checkout.is_dir()


# ── report ─────────────────────────────────────────────────


def _report() -> dict[str, object]:
    relay = t6_stats.RelayStatsCollector()
    relay.observe(_relay_stats_payload(relay_sse=90, event_batch=6, http_post=4))
    return t6.build_report(
        config={"users": 2, "sessions_per_user": 1, "turns_per_session": 2, "reply_words": 200},
        server={"ref": None, "commit": "1d5bd54", "base_url": "http://localhost:4321"},
        locust_exit_code=0,
        health=t6_stats.summarize_latency(
            [t6_stats.LatencySample(latency_ms=12.0, status_code=200)]
        ),
        policy=t6_stats.summarize_latency(
            [t6_stats.LatencySample(latency_ms=30.0, status_code=200)]
        ),
        event_loop={"stall_reports": 0, "max_stalled_s": 0.0},
        relay_counters=relay.summary(window_seconds=10.0),
        server_cpu={
            "cpu_seconds": 5.0,
            "cpu_seconds_per_1k_events": 0.05,
            "events_total": 100,
        },
        policy_session_id="conv_probe",
        artifacts={"server_log": "server-process.log"},
    )


def test_build_report_includes_required_sections() -> None:
    report = _report()

    assert report["scenario"] == "t6"
    assert report["health"]["p50_ms"] == 12.0
    assert report["policy_evaluate"]["p50_ms"] == 30.0
    assert report["event_loop"]["stall_reports"] == 0
    assert report["relay_counters"]["event_source_mix"]["relay_sse"] == 90
    assert report["server_cpu"]["cpu_seconds_per_1k_events"] == 0.05
    assert report["gate"]["pass"] is False  # fuse/overflow counters are unreported here


def test_gate_passes_with_healthy_numbers_and_zero_counters() -> None:
    report = _report()
    report["relay_counters"].update(  # type: ignore[union-attr]
        {
            "fuse_trips": 0,
            "fuse_trips_source": "counter",
            "subscriber_overflows": 0,
            "subscriber_overflows_source": "counter",
            "relay_persist_failures": 0,
            "relay_persist_failures_source": "counter",
        }
    )

    gate = t6.evaluate_gate(report)

    assert gate["pass"] is True
    assert gate["checks"]["health_p99_ms"]["status"] == "pass"


def test_gate_flags_stalls_and_slow_health() -> None:
    report = _report()
    report["event_loop"] = {"stall_reports": 1, "max_stalled_s": 6.5}
    report["health"] = t6_stats.summarize_latency(
        [t6_stats.LatencySample(latency_ms=6000.0, status_code=200)]
    ).as_dict()

    gate = t6.evaluate_gate(report)

    assert gate["pass"] is False
    assert gate["checks"]["no_watchdog_stalls"]["status"] == "fail"
    assert gate["checks"]["health_max_lt_1s"]["status"] == "fail"


def test_write_t6_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _report()

    json_path, markdown_path = t6.write_t6_report(tmp_path, report)

    assert json_path.name == "t6_report.json"
    assert markdown_path.name == "t6_report.md"
    assert json.loads(json_path.read_text())["scenario"] == "t6"
    text = markdown_path.read_text()
    for expected in (
        "T6 relay load report",
        "/health",
        "p50",
        "p99",
        "max",
        "watchdog",
        "fuse",
        "persistence",
        "subscriber overflow",
        "event source",
        "CPU / 1,000 events",
    ):
        assert expected in text
