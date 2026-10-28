# Omnigent load test

A load test where **each simulated user is a real Omnigent host**. Sibling to
`dev/benchmarks/`: benchmarks measure single-request latency in isolation; this
drives **concurrent, end-to-end load** — hosts → sessions → real multi-turn
conversations.

One unit of load = one **host**. Each Locust user spawns a real `omnigent host`
subprocess (unique identity), registers it over the host tunnel, then
repeatedly creates a **host-bound session** and drives a **real multi-turn
conversation** on it. Every turn is a genuine
`POST .../events` → server → the user's host → a runner subprocess it spawns →
LLM → stream → `idle` loop. The **LLM is mocked** (zero latency), so the numbers
isolate Omnigent's own dispatch / streaming / history-handling overhead rather
than provider latency. `-u N` scales the number of hosts.

## Capacity note (read this)

Turns really execute on the host, so each host spawns a **real runner
subprocess per session**, and each Locust user runs a **real host subprocess**.
`-u N` hosts × `--sessions-per-user M` therefore puts **N×M runner processes on
this (load-generating) machine**. This is deliberate — it drives genuine
end-to-end turns instead of faking the runner — but it is **capacity-limited**:
keep N modest (a few dozen). At high N the *load box*, not the server, saturates
first (Locust will warn about CPU). To stress the server harder, run this from a
beefier box or point the driver at a bigger server (see "Against another
server").

## Setup

Runs from a **repo checkout** (it imports `dev.benchmarks` + `tests`), with the
harness + bench deps:

```bash
uv sync --extra loadtest --extra agents-sdk
```

## Run

`run.py` boots the whole stack (server + zero-latency mock LLM), registers one
agent, then runs Locust against it. There is **no `--server` to pass** — mocking
the LLM requires a stack the test controls.

```bash
python dev/loadtest/run.py --users 8 --sessions-per-user 2 --turns-per-session 4
```

| Flag | Default | Meaning |
|---|---|---|
| `--users` | 4 | Concurrent hosts (N) — the main scale knob. |
| `--spawn-rate` | 1 | Hosts started per second. |
| `--run-time` | 120s | Run duration (`120s` / `5m` / `1h`). |
| `--sessions-per-user` | 2 | Host-bound sessions each host drives (sequentially). |
| `--turns-per-session` | 4 | Turns per session — history grows across them. |
| `--reply-words` | 60 | Word count of the mocked (streamed) reply per turn. |
| `--out-dir` | — | Result directory (default `results/omnigent_load_test-<timestamp>/`). |
| `--server-ref` | — | Run the server (and the hosts/runners driving it) from another checkout: a git ref like `v0.20.0`, or a directory path. |
| `--health-probe-interval` | 0.25 | Seconds between `GET /health` probes. |
| `--policy-probe-interval` | 1.0 | Seconds between `POST /v1/sessions/{id}/policies/evaluate` probes. |
| `--relay-stats-interval` | 5.0 | Seconds between the server's `relay_stats` log records. |
| `--hold-session-stream` | on | Hold each session's `GET /v1/sessions/{id}/stream` open while its turns run, like a real client. This is the only path that exercises per-subscriber SSE serialization; `--no-hold-session-stream` turns it off for comparison. |
| `--hold-stream-count` | 1 | Subscribers to hold per session. Raise it to multiply the server's per-event fan-out without adding runner processes; this is how the relay profile reaches saturation on one box. |

Start small to confirm the stack boots on your machine — `--users 2
--sessions-per-user 1 --turns-per-session 2 --run-time 40s` — then ramp.

## Results

Writes a timestamped `results/` directory (git-ignored):

| File | What it is |
|---|---|
| `summary.md` | Human-readable latency write-up — read this first. |
| `t6_report.md` | T6 relay health report — probe percentiles, relay counters, server CPU. |
| `t6_report.json` | Same report, machine-readable (used to compare runs). |
| `run_config.json` | Inputs + resolved locust argv + exit code. |
| `report_stats.csv` | Per-operation stats (raw). |
| `report_stats_history.csv` | Per-10s time series (raw). |
| `report_failures.csv` | Failure breakdown (raw). |
| `report.html` | Locust's own HTML report. |
| `console.log` | Full locust stdout/stderr. |
| `server-process.log` | Copy of the isolated server's process log (relay counters, watchdog reports). |

`summary.md` reports, per operation, count / failures / latency
(avg / median / p95 / p99 / max) + throughput. Key operations:

- **turn** — the headline: one full post→idle agent turn on a host's runner
  (mocked LLM). Latency **grows across a conversation** as history accumulates,
  so a rising p95/p99 with larger `--turns-per-session` is expected.
- **host online** — cost of a host subprocess registering over the host tunnel.
- **session create** — the host-bound `POST /v1/sessions`.
- **sse stream hold** — how long a user held its session's SSE stream open
  (a subscriber lifetime, not a request latency); the row's response length is
  the bytes that subscriber received. With `--hold-stream-count` above 1 each
  session contributes that many rows.
- **Ops/s** — aggregate throughput at this concurrency.

Each host's own log (useful when a host fails to register) is at
`results/.../host-workspaces/<host-name>/host.log`.

## Against another server

There is intentionally no `--server` flag: the mocked LLM only works on a stack
the test owns, and the whole point is real turns without token cost. To load a
different server you would need real runners + a real (or separately-mocked) LLM
on that side — out of scope here; use this to profile the host↔server↔runner
path locally.

To compare **checkouts** of this repo instead, pass `--server-ref`: the harness
creates a detached `git worktree` for the ref (or uses the directory you name),
pins `PYTHONPATH` so the server, the locust hosts, and the runners they spawn
import that checkout, and removes the worktree afterwards:

```bash
python dev/loadtest/run.py --users 2 --run-time 45s --server-ref v0.20.0 --out-dir /tmp/t6-baseline
```

`t6_report.json` records the ref and its commit, so two runs are comparable
side by side.

## T6 relay health

Every run is also a T6 run (`dev/loadtest/t6.py`). While locust drives load, the
harness probes the server and reads its Phase 0 relay counters:

- **Probes:** `GET /health` every 250 ms and
  `POST /v1/sessions/{id}/policies/evaluate` every 1 s against a dedicated
  unbound session. The report carries p50 / p99 / max latency and the non-200
  count for each.
- **Counters:** the server logs `relay_stats {json}` every
  `--relay-stats-interval` seconds. The harness tails that log record by record
  (it never imports `relay_stats`, which would read its own empty process) and
  reports the event-source mix, body-queue depth and byte peaks, subscriber
  serialization, relay stretch, and persistence failures.
- **Health signals:** event-loop watchdog stall reports (`Server event loop has
  not advanced for …`), body-queue fuse trips, and subscriber-stream overflow
  closes. Fuse and overflow counts come from the `relay_stats` payload when the
  build reports them, and from matching server-log lines otherwise.
- **CPU:** `psutil` samples the server PID; the report normalises measured CPU
  time to `cpu_seconds_per_1k_events`, the comparable figure across load levels
  and refs.
- **Gate:** pass requires health p99 < 250 ms, health max < 1 s, and zero
  watchdog stalls, fuse trips, persistence failures, and subscriber overflows.
  A signal the build does not report shows `unknown`, which fails the gate.

The server under test is fully isolated: `T6BenchEnvironment` gives it its own
`HOME`, `OMNIGENT_DATA_DIR`, `OMNIGENT_CONFIG_HOME`, and
`OMNIGENT_PROCESS_LOG_FILE` inside the run's temp dir, so no bench server writes
into `~/.omnigent/logs/server/`.

