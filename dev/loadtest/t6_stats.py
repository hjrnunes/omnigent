"""Parsing and aggregation helpers for the T6 relay load harness.

The T6 harness runs the server under test as a subprocess, so the Phase 0 relay
counters must be read from that process's log rather than imported in-process.
Everything here is pure (no server, no asyncio) so the parsing rules stay
unit-testable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELAY_STATS_LOG_EVENT = "relay_stats"
_RELAY_STATS_MARKER = f"{RELAY_STATS_LOG_EVENT} "
_RELAY_STATS_KEYS = ("window_seconds", "body_queue", "events", "relay_persist_failures")
_JSON_DECODER = json.JSONDecoder()

# The process-log formatter prints the message, not the ``extra`` fields, so
# match the rendered watchdog text and (defensively) the debug event name.
WATCHDOG_STALL_PATTERN = re.compile(
    r"Server event loop has not advanced for ([\d.]+)s|server_event_loop_stalled"
)
SUBSCRIBER_OVERFLOW_PATTERN = re.compile(r"subscriber overflowed", re.IGNORECASE)
FUSE_TRIP_PATTERN = re.compile(r"\bfuse\b[^\n]*\b(trip|tripped|exceed|abort|open)", re.IGNORECASE)
RELAY_PERSIST_FAILURE_PATTERN = re.compile(
    r"relay persist\w* fail|failed to persist relay", re.IGNORECASE
)

# Relay counters live at fixed payload paths. Read those paths only: a relay
# request id can contain "fuse" or "overflow", so matching on key text would
# count that request's depth as a guardrail trip.
_FUSE_TRIPS_PATH = ("body_queue", "fuse", "trips")
_SUBSCRIBER_OVERFLOWS_PATH = ("session_stream", "subscriber_overflows")


# ── relay_stats records ────────────────────────────────────


def parse_relay_stats_line(line: str) -> dict[str, Any] | None:
    """Return the ``relay_stats`` payload embedded in *line*, if any.

    The record can carry formatter attributes before or after the message, so
    every ``relay_stats `` occurrence is tried and the first JSON object that
    looks like a snapshot wins.
    """
    start = 0
    while True:
        index = line.find(_RELAY_STATS_MARKER, start)
        if index < 0:
            return None
        candidate = line[index + len(_RELAY_STATS_MARKER) :].lstrip()
        try:
            payload, _ = _JSON_DECODER.raw_decode(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and any(key in payload for key in _RELAY_STATS_KEYS):
            return payload
        start = index + len(_RELAY_STATS_MARKER)


@dataclass
class LogCounters:
    """Counters derived from whole server-log lines.

    The watchdog reports only appear as log lines. Fuse trips and subscriber
    overflows are the fallback for payloads that predate their counter fields,
    such as baseline v0.20.0 runs.
    """

    watchdog_stalls: int = 0
    max_stalled_s: float = 0.0
    fuse_log_trips: int = 0
    subscriber_overflow_logs: int = 0
    relay_persist_failure_logs: int = 0

    def observe(self, line: str) -> None:
        """Fold one log line into the counters."""
        watchdog = WATCHDOG_STALL_PATTERN.search(line)
        if watchdog is not None:
            self.watchdog_stalls += 1
            if watchdog.group(1) is not None:
                self.max_stalled_s = max(self.max_stalled_s, float(watchdog.group(1)))
        if SUBSCRIBER_OVERFLOW_PATTERN.search(line) is not None:
            self.subscriber_overflow_logs += 1
        if FUSE_TRIP_PATTERN.search(line) is not None:
            self.fuse_log_trips += 1
        if RELAY_PERSIST_FAILURE_PATTERN.search(line) is not None:
            self.relay_persist_failure_logs += 1


class RelayStatsCollector:
    """Keep the latest ``relay_stats`` snapshot and the peaks seen so far."""

    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None
        self._samples = 0
        self._peak: dict[str, int] = {
            "relay_persist_failures": 0,
            "fuse_trips": 0,
            "subscriber_overflows": 0,
            "outbound_max_depth": 0,
        }
        self._seen: dict[str, bool] = {
            "fuse_trips": False,
            "subscriber_overflows": False,
        }

    @property
    def samples(self) -> int:
        """Number of parsed snapshots."""
        return self._samples

    @property
    def latest(self) -> dict[str, Any] | None:
        """Most recent parsed payload, if any."""
        return self._latest

    def observe(self, payload: Mapping[str, Any]) -> None:
        """Record one snapshot payload."""
        self._latest = dict(payload)
        self._samples += 1
        persist = _as_mapping(payload.get("relay_persist_failures")).get("total")
        self._peak["relay_persist_failures"] = max(
            self._peak["relay_persist_failures"], _as_int(persist)
        )
        fuse = _counter_at_path(payload, _FUSE_TRIPS_PATH)
        if fuse is not None:
            self._seen["fuse_trips"] = True
            self._peak["fuse_trips"] = max(self._peak["fuse_trips"], fuse)
        overflow = _counter_at_path(payload, _SUBSCRIBER_OVERFLOWS_PATH)
        if overflow is not None:
            self._seen["subscriber_overflows"] = True
            self._peak["subscriber_overflows"] = max(self._peak["subscriber_overflows"], overflow)
        outbound = _as_mapping(payload.get("outbound_queue"))
        self._peak["outbound_max_depth"] = max(
            self._peak["outbound_max_depth"], _as_int(outbound.get("max_depth"))
        )

    def summary(
        self,
        *,
        window_seconds: float | None = None,
        log_counters: LogCounters | None = None,
    ) -> dict[str, Any]:
        """Return the report shape for the relay counters.

        Rates use *window_seconds* when provided; counts stay cumulative
        because the server's snapshot counters are cumulative for the process.
        """
        latest = self._latest or {}
        events = _as_mapping(latest.get("events"))
        body_queue = _as_mapping(latest.get("body_queue"))
        outbound_queue = _as_mapping(latest.get("outbound_queue"))
        serialization = _as_mapping(latest.get("subscriber_serialization"))
        stretch = _as_mapping(latest.get("relay_stretch"))

        by_source = {
            str(source): _as_int(count)
            for source, count in _as_mapping(events.get("by_source")).items()
        }
        total = sum(by_source.values())
        window = window_seconds if window_seconds and window_seconds > 0 else None
        by_type = {
            str(event_type): _as_int(_as_mapping(entry).get("count"))
            for event_type, entry in _as_mapping(events.get("by_type")).items()
        }

        fuse_reported = self._seen["fuse_trips"]
        overflow_reported = self._seen["subscriber_overflows"]
        fuse_source = "counter" if fuse_reported else "unreported"
        overflow_source = "counter" if overflow_reported else "unreported"
        fuse_value = self._peak["fuse_trips"]
        overflow_value = self._peak["subscriber_overflows"]
        if not fuse_reported and log_counters is not None:
            fuse_value = log_counters.fuse_log_trips
            fuse_source = "log"
        if not overflow_reported and log_counters is not None:
            overflow_value = log_counters.subscriber_overflow_logs
            overflow_source = "log"

        return {
            "available": self._latest is not None,
            "samples": self._samples,
            "events_total": total,
            "events_per_second": None if self._latest is None else _rate(total, window),
            "event_source_mix": dict(sorted(by_source.items())),
            "event_source_share_pct": (
                {
                    source: round(100.0 * count / total, 1)
                    for source, count in sorted(by_source.items())
                }
                if total
                else {}
            ),
            "event_source_per_second": {
                source: _rate(count, window) for source, count in sorted(by_source.items())
            },
            "event_type_mix": dict(sorted(by_type.items(), key=lambda item: -item[1])[:10]),
            "session_count": len(_as_mapping(events.get("sessions"))),
            "dropped_sessions": _as_int(events.get("dropped_sessions")),
            "dropped_event_types": _as_int(events.get("dropped_event_types")),
            "body_queue": {
                "max_depth": _as_int(body_queue.get("max_depth")),
                "max_bytes": _as_int(body_queue.get("max_bytes")),
                "requests_tracked": _as_int(body_queue.get("requests_tracked")),
                "dropped_requests": _as_int(body_queue.get("dropped_requests")),
                "warned_requests": _as_int(body_queue.get("warned_requests")),
                "fuse_threshold_bytes": _as_int(
                    _as_mapping(body_queue.get("fuse")).get("limit_bytes")
                ),
            },
            "outbound_queue": {
                "max_depth": self._peak["outbound_max_depth"],
                "runners_tracked": _as_int(outbound_queue.get("runners_tracked")),
                "dropped_samples": _as_int(outbound_queue.get("dropped_samples")),
            },
            "subscriber_serialization": {
                "count": _as_int(serialization.get("count")),
                "seconds_total": _as_float(serialization.get("seconds_total")),
                "seconds_max": _as_float(serialization.get("seconds_max")),
            },
            "relay_stretch": {
                "count": _as_int(stretch.get("count")),
                "seconds_total": _as_float(stretch.get("seconds_total")),
                "seconds_max": _as_float(stretch.get("seconds_max")),
            },
            "relay_persist_failures": self._peak["relay_persist_failures"],
            # No snapshot means the build has no Phase 0 counters at all, so a
            # zero here is "not reported" rather than "measured zero".
            "relay_persist_failures_source": (
                "counter" if self._latest is not None else "unreported"
            ),
            "fuse_trips": fuse_value,
            "fuse_trips_source": fuse_source,
            "subscriber_overflows": overflow_value,
            "subscriber_overflows_source": overflow_source,
        }


class ServerLogTail:
    """Incrementally read complete lines appended to the server process log."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._buffer = ""

    @property
    def path(self) -> Path:
        """Log file being watched."""
        return self._path

    def poll(self) -> list[str]:
        """Return complete lines added since the previous poll."""
        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except FileNotFoundError:
            return []
        if not chunk:
            return []
        text = self._buffer + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._buffer = lines.pop()
        return [line for line in lines if line]


# ── latency and CPU aggregation ────────────────────────────


@dataclass(frozen=True)
class LatencySample:
    """One probe attempt."""

    latency_ms: float
    status_code: int | None
    error: str | None = None


@dataclass(frozen=True)
class LatencySummary:
    """Percentile summary for one probe endpoint."""

    count: int
    ok: int
    non_200: int
    errors: int
    p50_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    mean_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return the report shape for this summary."""
        return {
            "count": self.count,
            "ok": self.ok,
            "non_200": self.non_200,
            "errors": self.errors,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
        }


def percentile_ms(values: Sequence[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile of *values* in milliseconds."""
    if not values:
        return None
    ordered = sorted(values)
    if fraction <= 0:
        return round(ordered[0], 1)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


def summarize_latency(samples: Sequence[LatencySample]) -> LatencySummary:
    """Aggregate probe samples into percentiles and status counts."""
    latencies = [sample.latency_ms for sample in samples]
    errors = sum(1 for sample in samples if sample.error is not None)
    return LatencySummary(
        count=len(samples),
        ok=sum(1 for sample in samples if sample.status_code == 200),
        non_200=sum(1 for sample in samples if sample.status_code != 200),
        errors=errors,
        p50_ms=percentile_ms(latencies, 0.50),
        p99_ms=percentile_ms(latencies, 0.99),
        max_ms=round(max(latencies), 1) if latencies else None,
        mean_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
    )


def cpu_seconds_per_1k_events(cpu_seconds: float | None, events_total: int | None) -> float | None:
    """Normalise measured CPU time to 1,000 relay events per second.

    ``cpu_seconds`` is the CPU time the server burned while the run sustained
    ``events_total / wall_seconds`` relay events per second, so the result is
    CPU seconds per 1,000 events/s of offered load.
    """
    if cpu_seconds is None or not events_total or events_total <= 0:
        return None
    return round(cpu_seconds / (events_total / 1000.0), 4)


# ── helpers ────────────────────────────────────────────────


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return round(float(value), 4)


def _rate(count: int, window_seconds: float | None) -> float | None:
    if window_seconds is None:
        return None
    return round(count / window_seconds, 2)


def _counter_at_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> int | None:
    """Return the numeric counter at exactly *path*, or ``None`` when absent.

    ``None`` tells the caller the build does not report that counter, so a
    reported zero stays distinct from a missing field.
    """
    node: object = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return int(node)
