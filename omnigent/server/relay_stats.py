"""Bounded, loop-confined counters for the relay measurement.

The counters answer one question: where does relay throughput go? They keep
per-request body-queue gauges and guardrails, per-runner outbound-queue gauges,
subscriber-overflow counts, per-session event rates by event type, the cost
of each subscriber's SSE serialization, the time the relay spends between
reader suspensions, the ingest mix, and silent ``_relay_persist`` failures.

Design constraints (see ``library/relay.md``):

- No locks and no threads: every recorder is a plain dict/float update that
  runs on the loop that owns the counter. Cross-loop body-queue callbacks are
  best-effort and may race on a counter; a lost sample never affects behavior.
- Every map is bounded, so a long soak cannot grow memory without limit. At a
  bound, the counter either evicts the oldest key or stops tracking the new
  key, and ``dropped_*`` fields make the tradeoff visible.
- Read through :func:`snapshot` in-process (unit tests, an in-process server), or
  read the periodic ``relay_stats`` log record from the server process log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from omnigent.debug_logging import debug_event

_logger = logging.getLogger(__name__)

RELAY_STATS_LOG_EVENT = "relay_stats"
"""Structured log record name emitted by the periodic task."""

RELAY_STATS_INTERVAL_ENV_VAR = "OMNIGENT_RELAY_STATS_INTERVAL_S"
"""Environment variable holding the periodic log interval in seconds."""

DEFAULT_RELAY_STATS_INTERVAL_S = 10.0
"""Default periodic log interval when the environment variable is unset."""

BODY_QUEUE_FUSE_ENV_VAR = "OMNIGENT_RELAY_BODY_QUEUE_FUSE_BYTES"
"""Environment variable holding the per-request body-queue fuse in bytes."""

DEFAULT_BODY_QUEUE_WARN_BYTES = 8 * 1024 * 1024
"""Per-request queued body bytes that log the first pressure warning."""

DEFAULT_BODY_QUEUE_FUSE_BYTES = 256 * 1024 * 1024
"""Per-request queued body bytes that abort that request by default."""


def body_queue_fuse_bytes() -> int:
    """Return the per-request body-queue fuse from the environment.

    Unset, non-numeric, and non-positive values fall back to
    :data:`DEFAULT_BODY_QUEUE_FUSE_BYTES`, so a typo cannot disable the fuse.
    """
    raw = os.environ.get(BODY_QUEUE_FUSE_ENV_VAR)
    if raw is None:
        return DEFAULT_BODY_QUEUE_FUSE_BYTES
    try:
        fuse_bytes = int(raw)
    except ValueError:
        return DEFAULT_BODY_QUEUE_FUSE_BYTES
    if fuse_bytes <= 0:
        return DEFAULT_BODY_QUEUE_FUSE_BYTES
    return fuse_bytes


SOURCE_RELAY_SSE = "relay_sse"
"""Events parsed from the runner's relay SSE stream."""

SOURCE_EVENT_BATCH = "event_batch"
"""Events delivered by a runner ``event.batch`` tunnel frame."""

SOURCE_HTTP_POST = "http_post"
"""Events posted to the HTTP ``POST /events`` route."""

_OTHER_EVENT_TYPE = "other"
"""Bucket that keeps per-session event-type maps bounded."""

_MAX_SESSIONS = 256
_MAX_EVENT_TYPES = 64
_MAX_EVENT_TYPES_PER_SESSION = 16
_MAX_TRACKED_REQUESTS = 512
_MAX_LOGGED_SESSIONS = 10
_MAX_LOGGED_REQUESTS = 10
_MAX_LOGGED_EVENT_TYPES = 25


@dataclass(frozen=True)
class BodyQueueSnapshot:
    """Current and peak tunnel body-queue pressure."""

    depth: int
    bytes: int
    max_depth: int
    max_bytes: int
    requests: dict[str, BodyQueueRequestSnapshot] = field(default_factory=dict)
    requests_tracked: int = 0
    dropped_requests: int = 0
    warned_requests: int = 0
    warn_bytes: int = 0
    fuse_trips: int = 0
    fuse_bytes: int = 0


@dataclass(frozen=True)
class BodyQueueRequestSnapshot:
    """Body frames queued for one tunneled request."""

    depth: int
    bytes: int
    warned: bool = False


@dataclass(frozen=True)
class BodyQueueGuard:
    """Guardrail outcome for one queued body frame.

    ``warn`` marks the single frame that crosses the warn threshold for its
    request. ``fuse_tripped`` marks every frame from the one that crosses the
    fuse, so the caller aborts the request exactly once.
    """

    request_bytes: int = 0
    warn_bytes: int = 0
    fuse_bytes: int = 0
    warn: bool = False
    fuse_tripped: bool = False


@dataclass(frozen=True)
class OutboundQueueSnapshot:
    """Current outbound WebSocket queue depth per runner and its peak."""

    depth: int = 0
    max_depth: int = 0
    runners: dict[str, int] = field(default_factory=dict)
    dropped_samples: int = 0

    @property
    def runners_tracked(self) -> int:
        """Number of runner queues the gauge currently tracks."""
        return len(self.runners)


@dataclass(frozen=True)
class DurationSnapshot:
    """Count and seconds for one timing counter."""

    count: int = 0
    seconds_total: float = 0.0
    seconds_max: float = 0.0

    @property
    def seconds_avg(self) -> float:
        """Mean seconds per sample, or ``0.0`` when there are no samples."""
        if self.count == 0:
            return 0.0
        return self.seconds_total / self.count


@dataclass(frozen=True)
class RelayStatsSnapshot:
    """Point-in-time view of every Phase 0 relay counter.

    Event-count fields are cumulative. ``*_per_second`` fields are rates over
    the window since the previous snapshot; calling :meth:`RelayStats.snapshot`
    starts a new window.
    """

    window_seconds: float
    body_queue: BodyQueueSnapshot
    events_by_source: dict[str, int]
    events_by_source_per_second: dict[str, float]
    events_by_type: dict[str, int]
    events_by_type_per_second: dict[str, float]
    events_by_session_by_type: dict[str, dict[str, int]]
    events_per_second_by_session_by_type: dict[str, dict[str, float]]
    subscriber_serialization_count: int
    subscriber_serialization_seconds_total: float
    subscriber_serialization_seconds_max: float
    subscriber_serialization_by_session: dict[str, DurationSnapshot]
    relay_stretch_count: int
    relay_stretch_seconds_total: float
    relay_stretch_seconds_max: float
    relay_stretch_by_session: dict[str, DurationSnapshot]
    relay_persist_failures: int
    relay_persist_failures_by_session: dict[str, int]
    outbound_queue: OutboundQueueSnapshot
    subscriber_overflows: int = 0
    dropped_sessions: int = 0
    dropped_event_types: int = 0


_T = TypeVar("_T")


class RelayStats:
    """Loop-confined relay counters with a bounded in-process snapshot."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = _MAX_SESSIONS,
        max_event_types: int = _MAX_EVENT_TYPES,
        max_event_types_per_session: int = _MAX_EVENT_TYPES_PER_SESSION,
        max_tracked_requests: int = _MAX_TRACKED_REQUESTS,
        warn_threshold_bytes: int | None = None,
        fuse_threshold_bytes: int | None = None,
    ) -> None:
        self._clock = clock
        self._max_sessions = max_sessions
        self._max_event_types = max_event_types
        self._max_event_types_per_session = max_event_types_per_session
        self._max_tracked_requests = max_tracked_requests
        self._warn_threshold_bytes = (
            DEFAULT_BODY_QUEUE_WARN_BYTES if warn_threshold_bytes is None else warn_threshold_bytes
        )
        self._fuse_threshold_bytes = (
            body_queue_fuse_bytes() if fuse_threshold_bytes is None else fuse_threshold_bytes
        )

        self._events_by_source: dict[str, int] = {}
        self._events_by_type: dict[str, int] = {}
        self._events_by_session_by_type: dict[str, dict[str, int]] = {}
        self._subscriber_serialization: dict[str, DurationSnapshot] = {}
        self._subscriber_serialization_count = 0
        self._subscriber_serialization_seconds_total = 0.0
        self._subscriber_serialization_seconds_max = 0.0
        self._relay_stretch: dict[str, DurationSnapshot] = {}
        self._relay_stretch_count = 0
        self._relay_stretch_seconds_total = 0.0
        self._relay_stretch_seconds_max = 0.0
        self._relay_persist_failures = 0
        self._relay_persist_failures_by_session: dict[str, int] = {}

        self._body_queue_depth = 0
        self._body_queue_bytes = 0
        self._body_queue_max_depth = 0
        self._body_queue_max_bytes = 0
        self._body_queue_requests: dict[str, BodyQueueRequestSnapshot] = {}
        # Rejected request ids stay here until their stream closes, so their
        # older uncounted frames can never consume a later counted frame.
        self._body_queue_rejected: dict[str, None] = {}
        self._body_queue_dropped_requests = 0
        self._body_queue_warned_requests = 0
        self._body_queue_fuse_trips = 0

        self._outbound_depths: dict[str, int] = {}
        self._outbound_max_depth = 0
        self._outbound_dropped_samples = 0
        self._subscriber_overflows = 0

        self._dropped_sessions = 0
        self._dropped_event_types = 0

        self._window_started_at = clock()
        self._window_events_by_source: dict[str, int] = {}
        self._window_events_by_type: dict[str, int] = {}
        self._window_events_by_session_by_type: dict[str, dict[str, int]] = {}

    # -- body queue -----------------------------------------------------

    def record_body_queue_enqueue(self, request_id: str, byte_count: int) -> BodyQueueGuard:
        """Record one body frame queued for ``request_id``.

        A request rejected at the tracking bound stays untracked until its
        stream closes, so frames queued before its rejection are never
        counted and can never consume a later counted frame's gauge. Such a
        request is also never guarded, matching its untracked byte total.

        :returns: The guardrail outcome for the frame. ``warn`` is ``True``
            for the single frame that first crosses the warn threshold, and
            ``fuse_tripped`` is ``True`` from the frame that crosses the fuse
            onward.
        """
        limit_bytes = self._fuse_threshold_bytes
        existing = self._body_queue_requests.get(request_id)
        if existing is None:
            if request_id in self._body_queue_rejected:
                self._body_queue_dropped_requests += 1
                return BodyQueueGuard(
                    warn_bytes=self._warn_threshold_bytes, fuse_bytes=limit_bytes
                )
            if len(self._body_queue_requests) >= self._max_tracked_requests:
                self._body_queue_dropped_requests += 1
                self._remember_rejected_request(request_id)
                return BodyQueueGuard(
                    warn_bytes=self._warn_threshold_bytes, fuse_bytes=limit_bytes
                )
            existing = BodyQueueRequestSnapshot(depth=0, bytes=0)
        request_bytes = existing.bytes + byte_count
        warn = request_bytes > self._warn_threshold_bytes and not existing.warned
        if warn:
            self._body_queue_warned_requests += 1
        fuse_tripped = request_bytes > limit_bytes
        if fuse_tripped:
            self._body_queue_fuse_trips += 1
        updated = BodyQueueRequestSnapshot(
            depth=existing.depth + 1,
            bytes=request_bytes,
            warned=existing.warned or warn,
        )
        self._body_queue_requests[request_id] = updated
        self._body_queue_depth += 1
        self._body_queue_bytes += byte_count
        self._body_queue_max_depth = max(self._body_queue_max_depth, self._body_queue_depth)
        self._body_queue_max_bytes = max(self._body_queue_max_bytes, self._body_queue_bytes)
        return BodyQueueGuard(
            request_bytes=request_bytes,
            warn_bytes=self._warn_threshold_bytes,
            fuse_bytes=limit_bytes,
            warn=warn,
            fuse_tripped=fuse_tripped,
        )

    def _remember_rejected_request(self, request_id: str) -> None:
        """Keep a rejected request untracked until its stream closes.

        The map holds one id per live rejected request, capped like the
        tracked map. Past the cap a rejection is still counted in
        ``dropped_requests`` but not remembered, so the gauge then only
        undercounts.
        """
        if len(self._body_queue_rejected) < self._max_tracked_requests:
            self._body_queue_rejected[request_id] = None

    def record_body_queue_dequeue(self, request_id: str, byte_count: int) -> None:
        """Record one body frame consumed from ``request_id``'s queue.

        A request rejected at the bound has no gauge, so its dequeues never
        move the aggregate: every frame in the tracked map is counted.
        """
        existing = self._body_queue_requests.get(request_id)
        if existing is None:
            return
        depth = max(existing.depth - 1, 0)
        byte_total = max(existing.bytes - byte_count, 0)
        self._body_queue_requests[request_id] = BodyQueueRequestSnapshot(
            depth=depth,
            bytes=byte_total,
            warned=existing.warned,
        )
        self._body_queue_depth -= existing.depth - depth
        self._body_queue_bytes -= existing.bytes - byte_total

    def record_body_queue_released(self, request_id: str) -> None:
        """Drop ``request_id``'s gauge when its stream closes or aborts.

        This also forgets a rejected id, so a request is only counted again
        after its stream closes, when its queue has drained.
        """
        self._body_queue_rejected.pop(request_id, None)
        existing = self._body_queue_requests.pop(request_id, None)
        if existing is None:
            return
        self._body_queue_depth = max(self._body_queue_depth - existing.depth, 0)
        self._body_queue_bytes = max(self._body_queue_bytes - existing.bytes, 0)

    # -- outbound queue -------------------------------------------------

    def record_outbound_queue_depth(self, runner_id: str, depth: int) -> bool:
        """Record one runner's outbound WebSocket queue depth.

        The runner map is bounded. A sample for a runner that does not fit is
        dropped and counted instead of evicting a live runner, so the peak
        keeps describing the runners already recorded.

        :returns: ``True`` when the sample was kept.
        """
        if runner_id not in self._outbound_depths:
            if len(self._outbound_depths) >= self._max_sessions:
                self._outbound_dropped_samples += 1
                return False
        depth = max(depth, 0)
        self._outbound_depths[runner_id] = depth
        self._outbound_max_depth = max(self._outbound_max_depth, depth)
        return True

    def record_outbound_queue_released(self, runner_id: str) -> None:
        """Drop a runner's outbound depth when its tunnel is replaced or closed."""
        self._outbound_depths.pop(runner_id, None)

    def record_subscriber_overflow(self) -> None:
        """Record one SSE subscriber the session stream dropped as too slow."""
        self._subscriber_overflows += 1

    # -- events ---------------------------------------------------------
    def record_event(self, session_id: str, event_type: str, *, source: str) -> None:
        """Record one event ingested from ``source`` for ``session_id``."""
        self._events_by_source[source] = self._events_by_source.get(source, 0) + 1
        self._bump(self._events_by_type, event_type, self._max_event_types)
        types = self._events_by_session_by_type.get(session_id)
        if types is None:
            if len(self._events_by_session_by_type) >= self._max_sessions:
                self._events_by_session_by_type.pop(next(iter(self._events_by_session_by_type)))
                self._dropped_sessions += 1
            types = {}
            self._events_by_session_by_type[session_id] = types
        self._bump(types, event_type, self._max_event_types_per_session)

        self._window_events_by_source[source] = self._window_events_by_source.get(source, 0) + 1
        self._bump(self._window_events_by_type, event_type, self._max_event_types)
        window_types = self._window_events_by_session_by_type.get(session_id)
        if window_types is None:
            if len(self._window_events_by_session_by_type) >= self._max_sessions:
                self._window_events_by_session_by_type.pop(
                    next(iter(self._window_events_by_session_by_type))
                )
            window_types = {}
            self._window_events_by_session_by_type[session_id] = window_types
        self._bump(window_types, event_type, self._max_event_types_per_session)

    def _bump(self, mapping: dict[str, int], event_type: str, limit: int) -> None:
        """Increment ``event_type`` in ``mapping``, folding overflow into ``other``."""
        if event_type in mapping:
            mapping[event_type] += 1
            return
        if len(mapping) < limit:
            mapping[event_type] = 1
            return
        self._dropped_event_types += 1
        mapping[_OTHER_EVENT_TYPE] = mapping.get(_OTHER_EVENT_TYPE, 0) + 1

    # -- timings --------------------------------------------------------

    def record_subscriber_serialization(self, session_id: str, seconds: float) -> None:
        """Record one subscriber's SSE serialization duration."""
        self._subscriber_serialization_count += 1
        self._subscriber_serialization_seconds_total += seconds
        self._subscriber_serialization_seconds_max = max(
            self._subscriber_serialization_seconds_max, seconds
        )
        existing = self._subscriber_serialization.get(session_id)
        if existing is None and len(self._subscriber_serialization) >= self._max_sessions:
            return
        self._subscriber_serialization[session_id] = _add_duration(existing, seconds)

    def record_relay_stretch(self, session_id: str, seconds: float) -> None:
        """Record one stretch between two suspensions of a relay reader."""
        self._relay_stretch_count += 1
        self._relay_stretch_seconds_total += seconds
        self._relay_stretch_seconds_max = max(self._relay_stretch_seconds_max, seconds)
        existing = self._relay_stretch.get(session_id)
        if existing is None and len(self._relay_stretch) >= self._max_sessions:
            return
        self._relay_stretch[session_id] = _add_duration(existing, seconds)

    # -- persist --------------------------------------------------------

    def record_relay_persist_failure(self, session_id: str) -> None:
        """Record one swallowed ``_relay_persist`` failure."""
        self._relay_persist_failures += 1
        existing = self._relay_persist_failures_by_session.get(session_id)
        if existing is None and len(self._relay_persist_failures_by_session) >= self._max_sessions:
            return
        self._relay_persist_failures_by_session[session_id] = (existing or 0) + 1

    # -- snapshot -------------------------------------------------------

    def snapshot(self) -> RelayStatsSnapshot:
        """Return a frozen snapshot and start a new rate window."""
        now = self._clock()
        window_seconds = max(now - self._window_started_at, 0.0)

        events_by_source = dict(self._events_by_source)
        events_by_type = dict(self._events_by_type)
        events_by_session_by_type = {
            session_id: dict(types)
            for session_id, types in self._events_by_session_by_type.items()
        }
        window_by_source = self._window_events_by_source
        window_by_type = self._window_events_by_type
        window_by_session = self._window_events_by_session_by_type

        self._window_started_at = now
        self._window_events_by_source = {}
        self._window_events_by_type = {}
        self._window_events_by_session_by_type = {}

        def rate(count: int) -> float:
            if window_seconds <= 0.0:
                return 0.0
            return count / window_seconds

        return RelayStatsSnapshot(
            window_seconds=window_seconds,
            body_queue=BodyQueueSnapshot(
                depth=self._body_queue_depth,
                bytes=self._body_queue_bytes,
                max_depth=self._body_queue_max_depth,
                max_bytes=self._body_queue_max_bytes,
                requests=dict(self._body_queue_requests),
                requests_tracked=len(self._body_queue_requests),
                dropped_requests=self._body_queue_dropped_requests,
                warned_requests=self._body_queue_warned_requests,
                warn_bytes=self._warn_threshold_bytes,
                fuse_trips=self._body_queue_fuse_trips,
                fuse_bytes=self._fuse_threshold_bytes,
            ),
            events_by_source=events_by_source,
            events_by_source_per_second={
                source: rate(count) for source, count in window_by_source.items()
            },
            events_by_type=events_by_type,
            events_by_type_per_second={
                event_type: rate(count) for event_type, count in window_by_type.items()
            },
            events_by_session_by_type=events_by_session_by_type,
            events_per_second_by_session_by_type={
                session_id: {event_type: rate(count) for event_type, count in types.items()}
                for session_id, types in window_by_session.items()
            },
            subscriber_serialization_count=self._subscriber_serialization_count,
            subscriber_serialization_seconds_total=(self._subscriber_serialization_seconds_total),
            subscriber_serialization_seconds_max=self._subscriber_serialization_seconds_max,
            subscriber_serialization_by_session=dict(self._subscriber_serialization),
            relay_stretch_count=self._relay_stretch_count,
            relay_stretch_seconds_total=self._relay_stretch_seconds_total,
            relay_stretch_seconds_max=self._relay_stretch_seconds_max,
            relay_stretch_by_session=dict(self._relay_stretch),
            relay_persist_failures=self._relay_persist_failures,
            relay_persist_failures_by_session=dict(self._relay_persist_failures_by_session),
            outbound_queue=OutboundQueueSnapshot(
                depth=sum(self._outbound_depths.values()),
                max_depth=self._outbound_max_depth,
                runners=dict(self._outbound_depths),
                dropped_samples=self._outbound_dropped_samples,
            ),
            subscriber_overflows=self._subscriber_overflows,
            dropped_sessions=self._dropped_sessions,
            dropped_event_types=self._dropped_event_types,
        )


BODY_WAIT_PROBE_EXTENSION_KEY = "omnigent.body_wait_probe"
"""``httpx.Response.extensions`` key carrying a tunneled body's wait probe."""


@dataclass
class BodyWaitProbe:
    """Marks each time a tunneled body read suspends on an empty queue."""

    wait_count: int = 0
    wait_started_at: float = 0.0

    def note_wait_started(self, timestamp: float) -> None:
        """Record a body-queue wait starting at ``timestamp``."""
        self.wait_count += 1
        self.wait_started_at = timestamp


class RelayStretchTimer:
    """Times relay loop stretches between real suspensions of one reader.

    With a probe, a stretch spans chunks that were already queued and ends
    where a body-queue wait started, including a wait that ends the stream.
    Without one, each chunk is its own stretch.
    """

    def __init__(
        self,
        session_id: str,
        *,
        body_wait_probe: BodyWaitProbe | None = None,
        clock: Callable[[], float] = time.perf_counter,
        record: Callable[[str, float], None] | None = None,
    ) -> None:
        self._session_id = session_id
        self._probe = body_wait_probe
        self._clock = clock
        # Resolved lazily because the module recorder is defined below.
        self._record = record_relay_stretch if record is None else record
        self._seen_wait_count = 0 if body_wait_probe is None else body_wait_probe.wait_count
        self._started_at: float | None = None if body_wait_probe is None else clock()

    def chunk_received(self) -> None:
        """Note that the reader resumed with a chunk."""
        probe = self._probe
        if probe is not None and probe.wait_count != self._seen_wait_count:
            self._seen_wait_count = probe.wait_count
            self._end(probe.wait_started_at)
        if probe is None or self._started_at is None:
            self._started_at = self._clock()

    def chunk_processed(self) -> None:
        """Note that the reader finished a chunk and is about to read again."""
        if self._probe is None:
            self._end(self._clock())

    async def bound(self, awaitable: Awaitable[_T]) -> _T:
        """Await ``awaitable`` outside of any stretch."""
        self._end(self._clock())
        try:
            return await awaitable
        finally:
            self._started_at = self._clock()

    def finish(self) -> None:
        """Record the open stretch, honoring a body wait that ended the stream."""
        probe = self._probe
        if probe is not None and probe.wait_count != self._seen_wait_count:
            # The read never resumed: EOF, cancellation, timeout, or a
            # disconnect ended a suspended body wait.
            self._seen_wait_count = probe.wait_count
            self._end(probe.wait_started_at)
            return
        self._end(self._clock())

    def _end(self, ended_at: float) -> None:
        """Record the open stretch as ending at ``ended_at``."""
        started_at = self._started_at
        if started_at is None:
            return
        self._started_at = None
        self._record(self._session_id, max(ended_at - started_at, 0.0))


def body_wait_probe_for_response(response: object) -> BodyWaitProbe | None:
    """Return the body-wait probe a response carries, or ``None``."""
    extensions = getattr(response, "extensions", None)
    if not isinstance(extensions, dict):
        return None
    probe = extensions.get(BODY_WAIT_PROBE_EXTENSION_KEY)
    return probe if isinstance(probe, BodyWaitProbe) else None


def _add_duration(
    existing: DurationSnapshot | None,
    seconds: float,
) -> DurationSnapshot:
    """Accumulate one timing sample for one session."""
    if existing is None:
        return DurationSnapshot(count=1, seconds_total=seconds, seconds_max=seconds)
    return DurationSnapshot(
        count=existing.count + 1,
        seconds_total=existing.seconds_total + seconds,
        seconds_max=max(existing.seconds_max, seconds),
    )


_default_stats = RelayStats()


def snapshot() -> RelayStatsSnapshot:
    """Return the process-wide relay counters and start a new rate window."""
    return _default_stats.snapshot()


def reset_for_tests() -> None:
    """Reset the process-wide counters (tests only)."""
    global _default_stats
    _default_stats = RelayStats()


def body_byte_length(body: str) -> int:
    """Return the UTF-8 byte length of a frame body string.

    ``ResponseBodyFrame.body`` carries the payload as a ``utf-8`` string or as
    ASCII ``base64``, so this one measure counts a queued body's real bytes.
    """
    return len(body.encode("utf-8"))


def dequeued_body_byte_length(body: str, encoding: str, decoded: bytes) -> int:
    """Return a consumed frame's byte count without re-encoding its body.

    :param body: The frame's body string.
    :param encoding: The frame's declared encoding.
    :param decoded: The body returned by ``decode_body``, reused when the
        encoding already produced the byte count.
    """
    if encoding == "utf-8":
        return len(decoded)
    return len(body)


def record_body_queue_enqueue(request_id: str, byte_count: int) -> BodyQueueGuard:
    """Record a queued tunnel body frame and return its guardrail outcome."""
    return _default_stats.record_body_queue_enqueue(request_id, byte_count)


def record_body_queue_dequeue(request_id: str, byte_count: int) -> None:
    """Record a consumed tunnel body frame for ``request_id``."""
    _default_stats.record_body_queue_dequeue(request_id, byte_count)


def record_body_queue_released(request_id: str) -> None:
    """Drop the gauge for a tunneled request that closed."""
    _default_stats.record_body_queue_released(request_id)


def record_outbound_queue_depth(runner_id: str, depth: int) -> bool:
    """Record one runner's outbound queue depth; ``False`` when not kept."""
    return _default_stats.record_outbound_queue_depth(runner_id, depth)


def record_outbound_queue_released(runner_id: str) -> None:
    """Drop a runner's outbound queue depth when its tunnel closes."""
    _default_stats.record_outbound_queue_released(runner_id)


def record_subscriber_overflow() -> None:
    """Record one SSE subscriber dropped as too slow."""
    _default_stats.record_subscriber_overflow()


def record_event(session_id: str, event_type: str, *, source: str) -> None:
    """Record one ingested event and its source."""
    _default_stats.record_event(session_id, event_type, source=source)


def record_subscriber_serialization(session_id: str, seconds: float) -> None:
    """Record one subscriber's SSE serialization duration."""
    _default_stats.record_subscriber_serialization(session_id, seconds)


def record_relay_stretch(session_id: str, seconds: float) -> None:
    """Record one relay stretch between reader suspensions."""
    _default_stats.record_relay_stretch(session_id, seconds)


def record_relay_persist_failure(session_id: str) -> None:
    """Record one swallowed ``_relay_persist`` failure."""
    _default_stats.record_relay_persist_failure(session_id)


def _busiest(mapping: dict[str, _T], *, key: Callable[[_T], float], limit: int) -> dict[str, _T]:
    """Return at most ``limit`` entries ordered by ``key`` descending."""
    ordered = sorted(mapping.items(), key=lambda item: key(item[1]), reverse=True)
    return dict(ordered[:limit])


def build_relay_stats_log_payload(
    stats_snapshot: RelayStatsSnapshot,
    *,
    max_sessions: int | None = None,
    max_requests: int = _MAX_LOGGED_REQUESTS,
    max_event_types: int = _MAX_LOGGED_EVENT_TYPES,
) -> dict[str, Any]:
    """Build the bounded JSON payload for the periodic ``relay_stats`` record."""
    session_limit = _MAX_LOGGED_SESSIONS if max_sessions is None else max_sessions
    body_queue = stats_snapshot.body_queue
    busy_sessions = _busiest(
        stats_snapshot.events_by_session_by_type,
        key=lambda types: float(sum(types.values())),
        limit=session_limit,
    )
    return {
        "window_seconds": round(stats_snapshot.window_seconds, 3),
        "body_queue": {
            "depth": body_queue.depth,
            "bytes": body_queue.bytes,
            "max_depth": body_queue.max_depth,
            "max_bytes": body_queue.max_bytes,
            "requests_tracked": body_queue.requests_tracked,
            "dropped_requests": body_queue.dropped_requests,
            "warned_requests": body_queue.warned_requests,
            "warn_bytes": body_queue.warn_bytes,
            "fuse": {
                "trips": body_queue.fuse_trips,
                "limit_bytes": body_queue.fuse_bytes,
            },
            "requests": {
                request_id: {
                    "depth": request.depth,
                    "bytes": request.bytes,
                    "warned": request.warned,
                }
                for request_id, request in _busiest(
                    body_queue.requests,
                    key=lambda request: request.depth,
                    limit=max_requests,
                ).items()
            },
        },
        "outbound_queue": {
            "depth": stats_snapshot.outbound_queue.depth,
            "max_depth": stats_snapshot.outbound_queue.max_depth,
            "runners_tracked": stats_snapshot.outbound_queue.runners_tracked,
            "dropped_samples": stats_snapshot.outbound_queue.dropped_samples,
            "runners": _busiest(
                stats_snapshot.outbound_queue.runners,
                key=float,
                limit=max_requests,
            ),
        },
        "session_stream": {
            "subscriber_overflows": stats_snapshot.subscriber_overflows,
        },
        "events": {
            "by_source": dict(stats_snapshot.events_by_source),
            "by_source_per_second": {
                source: round(rate, 3)
                for source, rate in stats_snapshot.events_by_source_per_second.items()
            },
            "by_type": {
                event_type: {
                    "count": count,
                    "per_second": round(
                        stats_snapshot.events_by_type_per_second.get(event_type, 0.0), 3
                    ),
                }
                for event_type, count in _busiest(
                    stats_snapshot.events_by_type,
                    key=lambda count: float(count),
                    limit=max_event_types,
                ).items()
            },
            "sessions": {
                session_id: {
                    "total": sum(types.values()),
                    "per_second": round(
                        sum(
                            stats_snapshot.events_per_second_by_session_by_type.get(
                                session_id, {}
                            ).values()
                        ),
                        3,
                    ),
                    "types": {
                        event_type: {
                            "count": count,
                            "per_second": round(
                                stats_snapshot.events_per_second_by_session_by_type.get(
                                    session_id, {}
                                ).get(event_type, 0.0),
                                3,
                            ),
                        }
                        for event_type, count in types.items()
                    },
                }
                for session_id, types in busy_sessions.items()
            },
            "dropped_sessions": stats_snapshot.dropped_sessions,
            "dropped_event_types": stats_snapshot.dropped_event_types,
        },
        "subscriber_serialization": {
            "count": stats_snapshot.subscriber_serialization_count,
            "seconds_total": round(stats_snapshot.subscriber_serialization_seconds_total, 6),
            "seconds_max": round(stats_snapshot.subscriber_serialization_seconds_max, 6),
            "sessions": _duration_payload(
                stats_snapshot.subscriber_serialization_by_session, session_limit
            ),
        },
        "relay_stretch": {
            "count": stats_snapshot.relay_stretch_count,
            "seconds_total": round(stats_snapshot.relay_stretch_seconds_total, 6),
            "seconds_max": round(stats_snapshot.relay_stretch_seconds_max, 6),
            "sessions": _duration_payload(stats_snapshot.relay_stretch_by_session, session_limit),
        },
        "relay_persist_failures": {
            "total": stats_snapshot.relay_persist_failures,
            "sessions": dict(
                sorted(
                    stats_snapshot.relay_persist_failures_by_session.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:session_limit]
            ),
        },
    }


def _duration_payload(
    mapping: dict[str, DurationSnapshot],
    limit: int,
) -> dict[str, dict[str, Any]]:
    """Render the busiest timing sessions for the log payload."""
    return {
        session_id: {
            "count": duration.count,
            "seconds_total": round(duration.seconds_total, 6),
            "seconds_max": round(duration.seconds_max, 6),
            "seconds_avg": round(duration.seconds_avg, 6),
        }
        for session_id, duration in _busiest(
            mapping,
            key=lambda duration: duration.seconds_total,
            limit=limit,
        ).items()
    }


def log_relay_stats_snapshot(stats_snapshot: RelayStatsSnapshot | None = None) -> None:
    """Emit one structured ``relay_stats`` log record."""
    current = snapshot() if stats_snapshot is None else stats_snapshot
    payload = build_relay_stats_log_payload(current)
    _logger.info(
        "%s %s",
        RELAY_STATS_LOG_EVENT,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        extra=debug_event(
            RELAY_STATS_LOG_EVENT,
            window_seconds=current.window_seconds,
            body_queue_depth=current.body_queue.depth,
            body_queue_bytes=current.body_queue.bytes,
            body_queue_warned_requests=current.body_queue.warned_requests,
            body_queue_fuse_trips=current.body_queue.fuse_trips,
            outbound_queue_max_depth=current.outbound_queue.max_depth,
            subscriber_overflows=current.subscriber_overflows,
            events_relay_sse=current.events_by_source.get(SOURCE_RELAY_SSE, 0),
            events_event_batch=current.events_by_source.get(SOURCE_EVENT_BATCH, 0),
            events_http_post=current.events_by_source.get(SOURCE_HTTP_POST, 0),
            subscriber_serializations=current.subscriber_serialization_count,
            subscriber_serialization_seconds_total=(
                current.subscriber_serialization_seconds_total
            ),
            relay_stretches=current.relay_stretch_count,
            relay_stretch_seconds_total=current.relay_stretch_seconds_total,
            relay_persist_failures=current.relay_persist_failures,
        ),
    )


def relay_stats_interval_seconds() -> float | None:
    """Return the configured log interval, or ``None`` when logging is disabled."""
    raw = os.environ.get(RELAY_STATS_INTERVAL_ENV_VAR)
    if raw is None:
        return DEFAULT_RELAY_STATS_INTERVAL_S
    try:
        interval = float(raw)
    except ValueError:
        return DEFAULT_RELAY_STATS_INTERVAL_S
    if not math.isfinite(interval) or interval <= 0.0:
        return None
    return interval


async def log_relay_stats_periodically(interval_seconds: float | None = None) -> None:
    """Emit a ``relay_stats`` record every ``interval_seconds`` until cancelled."""
    interval = relay_stats_interval_seconds() if interval_seconds is None else interval_seconds
    if interval is None or interval <= 0.0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            log_relay_stats_snapshot()
        except Exception:
            _logger.exception("relay stats snapshot failed")
