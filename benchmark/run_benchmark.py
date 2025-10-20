
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shlex
import subprocess
import sys
import time
import uuid
import threading
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_CLIENTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_SCALES = [10, 100]
DEFAULT_EXPLAIN_QUERY = "SELECT abalance FROM pgbench_accounts WHERE aid = 1"


REPO_ROOT = Path(__file__).resolve().parent.parent


class CommandError(RuntimeError):
    """Raised when a shell command exits with a non-zero status."""


def run_command(cmd: List[str], *, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess:
    """Execute *cmd* and return the completed process.

    Parameters
    ----------
    cmd:
        The command to execute. Each argument must already be shell-escaped.
    check:
        When ``True`` (the default) raise :class:`CommandError` if the process
        exits with a non-zero status code.
    input_text:
        Optional text to feed to the command's standard input. The input is
        encoded as UTF-8.
    """

    result = subprocess.run(
        cmd,
        input=input_text,   # <-- remove .encode()
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check and result.returncode != 0:
        raise CommandError(
            f"Command {' '.join(cmd)} failed with exit code {result.returncode}\n{result.stdout}"
        )
    if result.stdout:
        sys.stdout.write(result.stdout)
    return result


def stream_command(cmd: List[str]) -> str:
    """Execute *cmd* while streaming stdout/stderr to the caller."""

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None  # For type checkers
    output_lines: List[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise CommandError(f"Command {' '.join(cmd)} failed with exit code {return_code}")
    return "".join(output_lines)

def _ssh_cmd(host: str, ssh_binary: str, ssh_user: str | None, ssh_opts: str) -> List[str]:
    # Build the ssh prefix list (supports spaces in --ssh-binary)
    base = shlex.split(ssh_binary)
    if ssh_user:
        dest = f"{ssh_user}@{host}"
    else:
        dest = host
    return base + shlex.split(ssh_opts) + [dest, "--"]

def _read_remote(host: str, cmd: str, *, ssh_binary: str, ssh_user: str | None, ssh_opts: str) -> str:
    full = _ssh_cmd(host, ssh_binary, ssh_user, ssh_opts) + [cmd]
    result = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise CommandError(f"SSH to {host!r} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout

def _read_proc_stat_line(host: str, ssh_binary: str, ssh_user: str | None, ssh_opts: str) -> list[int]:
    # Returns jiffy counters from the "cpu" summary line
    out = _read_remote(host, "cat /proc/stat | head -n1", ssh_binary=ssh_binary, ssh_user=ssh_user, ssh_opts=ssh_opts)
    # Format: cpu  user nice system idle iowait irq softirq steal guest guest_nice
    parts = out.strip().split()
    if not parts or parts[0] != "cpu":
        raise CommandError(f"Unexpected /proc/stat format from {host}: {out!r}")
    return [int(x) for x in parts[1:11]]  # first 10 counters is plenty

def _cpu_percent_from_two_reads(prev: list[int], curr: list[int]) -> float:
    # Compute CPU% = 100 * (1 - idle_delta/total_delta)
    idle_prev = prev[3] + prev[4]  # idle + iowait
    idle_curr = curr[3] + curr[4]
    total_prev = sum(prev)
    total_curr = sum(curr)
    idle_delta = idle_curr - idle_prev
    total_delta = total_curr - total_prev
    if total_delta <= 0:
        return 0.0
    usage = 100.0 * (1.0 - (idle_delta / total_delta))
    # Clamp for safety
    return max(0.0, min(100.0, usage))

def _mem_percent(host: str, ssh_binary: str, ssh_user: str | None, ssh_opts: str) -> float:
    out = _read_remote(host, "cat /proc/meminfo", ssh_binary=ssh_binary, ssh_user=ssh_user, ssh_opts=ssh_opts)
    total = avail = None
    for line in out.splitlines():
        if line.startswith("MemTotal:"):
            total = float(line.split()[1])  # kB
        elif line.startswith("MemAvailable:"):
            avail = float(line.split()[1])  # kB
        if total is not None and avail is not None:
            break
    if not total or not avail:
        return float("nan")
    used_pct = 100.0 * (1.0 - (avail / total))
    return max(0.0, min(100.0, used_pct))

def percentile(values: Sequence[float], fraction: float) -> float:
    """Compute the percentile defined by *fraction* (0.0–1.0) for *values*."""

    if not values:
        return float("nan")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]

    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def build_psql_command(*, host: str, port: int, database: str, user: str, extra_args: Sequence[str] | None = None) -> List[str]:
    cmd = ["psql", "-v", "ON_ERROR_STOP=1", "-h", host, "-p", str(port), "-U", user, "-d", database]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def ensure_pg_stat_statements(*, host: str, port: int, database: str, user: str) -> None:
    run_command(
        build_psql_command(host=host, port=port, database=database, user=user),
        input_text="CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
    )


def reset_pg_stat_statements(*, host: str, port: int, database: str, user: str) -> None:
    run_command(
        build_psql_command(host=host, port=port, database=database, user=user),
        input_text="SELECT pg_stat_statements_reset();",
    )


def collect_pg_stat_statements(
        *, host: str, port: int, database: str, user: str, limit: int = 10
) -> List[dict[str, str]]:
    sql = " ".join([
        "SELECT queryid, calls, mean_time, rows, shared_blks_hit, shared_blks_read,",
        "shared_blks_dirtied, shared_blks_written, temp_blks_read, temp_blks_written,",
        "blk_read_time, blk_write_time, query",
        "FROM pg_stat_statements",
        "WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())",
        f"ORDER BY mean_time DESC LIMIT {limit}",
    ])
    result = subprocess.run(
        build_psql_command(host=host, port=port, database=database, user=user, extra_args=["--csv", "-c", sql]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise CommandError(f"Failed to collect pg_stat_statements: {message}")
    reader = csv.DictReader(io.StringIO(result.stdout))
    return list(reader)


def _normalize_query_for_explain(query: str) -> str | None:
    query = query.strip()
    if not query:
        return None

    first_statement = query.split(";")[0].strip()
    if not first_statement:
        return None

    allowed_prefixes = ("SELECT", "UPDATE", "INSERT", "DELETE")
    if not first_statement.upper().startswith(allowed_prefixes):
        return None

    normalized = re.sub(r":[a-zA-Z_][a-zA-Z0-9_]*", "1", first_statement)
    return normalized


def select_query_for_explain(rows: Sequence[dict[str, str]]) -> str | None:
    """Choose a representative query from pg_stat_statements for EXPLAIN."""

    for row in rows:
        candidate = row.get("query", "")
        normalized = _normalize_query_for_explain(candidate)
        if normalized:
            return normalized
    return None


def capture_explain_plan(
        *, host: str, port: int, database: str, user: str, query: str, destination: Path
) -> None:
    sql = "\n".join([
        "SET client_min_messages TO WARNING;",
        "BEGIN;",
        f"EXPLAIN (ANALYZE, BUFFERS) {query};",
        "ROLLBACK;",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_command(
            build_psql_command(host=host, port=port, database=database, user=user),
            input_text=sql,
            check=False,
        )
        output = result.stdout or ""
        if result.returncode != 0:
            output = f"Failed to capture plan (exit code {result.returncode}):\n{output}"
        destination.write_text(output, encoding="utf-8")
    except CommandError as exc:
        destination.write_text(str(exc), encoding="utf-8")


def drop_pgbench_tables(*, host: str, port: int, database: str, user: str) -> None:
    sql = "\n".join([
        "DROP TABLE IF EXISTS pgbench_history CASCADE;",
        "DROP TABLE IF EXISTS pgbench_tellers CASCADE;",
        "DROP TABLE IF EXISTS pgbench_accounts CASCADE;",
        "DROP TABLE IF EXISTS pgbench_branches CASCADE;",
    ])
    run_command(build_psql_command(host=host, port=port, database=database, user=user), input_text=sql)


def fetch_active_workers(*, host: str, port: int, database: str, user: str) -> List[tuple[str, int]]:
    result = subprocess.run(
        build_psql_command(host=host, port=port, database=database, user=user, extra_args=[
            "-At", "-c", "SELECT node_name, node_port FROM master_get_active_worker_nodes();"
        ]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    workers: List[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        name, port_str = line.split("|")
        workers.append((name, int(port_str)))
    return workers

def set_shard_count(*, host: str, port: int, database: str, user: str, shards: int) -> None:
    run_command(
        build_psql_command(host=host, port=port, database=database, user=user),
        input_text=f"SET citus.shard_count = {int(shards)};",
    )

def synchronize_workers(
        workers: Iterable[str], *, host: str, port: int, database: str, user: str, worker_port: int
) -> None:
    desired = list(workers)
    desired_set = set(desired)
    active = fetch_active_workers(host=host, port=port, database=database, user=user)

    # Remove extras
    removal_statements = [
        f"SELECT master_remove_node('{node_name}', {node_port});"
        for (node_name, node_port) in active if node_name not in desired_set
    ]
    if removal_statements:
        run_command(build_psql_command(host=host, port=port, database=database, user=user),
                    input_text="\n".join(removal_statements))

    # Add missing
    statements = ["CREATE EXTENSION IF NOT EXISTS citus;"]
    for worker in desired:
        statements.append(
            "DO $$\nBEGIN\n"
            f"    IF NOT EXISTS (\n"
            f"        SELECT 1 FROM master_get_active_worker_nodes()\n"
            f"        WHERE node_name = '{worker}' AND node_port = {worker_port}\n"
            f"    ) THEN\n"
            f"        PERFORM master_add_node('{worker}', {worker_port});\n"
            f"    END IF;\n"
            "END;\n$$;"
        )
    run_command(build_psql_command(host=host, port=port, database=database, user=user),
                input_text="\n".join(statements))

def create_pgbench_schema(*, host: str, port: int, database: str, user: str) -> None:
    # Create empty pgbench tables + primary keys; no data generated
    run_command([
        "pgbench",
        "-h", host, "-p", str(port),
        "-i",
        "-I", "dtp",   # d=drop, t=create tables, p=create primary keys
        "-U", user,
        database,
    ])


def distribute_pgbench_tables(*, host: str, port: int, database: str, user: str) -> None:
    sql = "\n".join([
        "CREATE EXTENSION IF NOT EXISTS citus;",
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = 'pgbench_accounts'::regclass) THEN "
        "PERFORM create_distributed_table('pgbench_accounts', 'aid'); END IF; END; $$;",
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = 'pgbench_branches'::regclass AND partmethod = 'n') THEN "
        "PERFORM create_reference_table('pgbench_branches'); END IF; END; $$;",
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = 'pgbench_tellers'::regclass AND partmethod = 'n') THEN "
        "PERFORM create_reference_table('pgbench_tellers'); END IF; END; $$;",
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_dist_partition WHERE logicalrelid = 'pgbench_history'::regclass) THEN "
        "PERFORM create_distributed_table('pgbench_history', 'tid'); END IF; END; $$;",
    ])
    run_command(build_psql_command(host=host, port=port, database=database, user=user), input_text=sql)

def load_pgbench_data(*, host: str, port: int, database: str, user: str, scale: int) -> None:
    # Generate data + vacuum; DO NOT recreate/drop tables
    run_command([
        "pgbench",
        "-h", host, "-p", str(port),
        "-i",
        "-I", "gv",   # g=generate data, v=vacuum
        "-n",         # no-vacuum-freeze/skip recreating objects (operate on existing)
        "-s", str(scale),
        "-U", user,
        database,
    ])

def initialize_pgbench_dataset(*, host: str, port: int, database: str, scale: int, user: str) -> None:
    run_command([
        "pgbench",
        "-h", host,
        "-p", str(port),
        "-i",
        "-I", "dtgvp",
        "-s", str(scale),
        "-U", user,
        database,
    ])


def parse_clients(arg: List[str]) -> List[int]:
    if not arg:
        return DEFAULT_CLIENTS
    clients: List[int] = []
    for item in arg:
        clients.append(int(item))
    if not clients:
        raise argparse.ArgumentTypeError("At least one client concurrency must be supplied")
    return clients


def parse_worker_groups(arg: List[str] | None) -> List[tuple[str, ...]]:
    """
    Parse one or more --worker-group arguments into a list of groups,
    where each group is a tuple of worker hostnames/IPs.

    Examples:
      --worker-group 10.0.0.11,10.0.0.12,10.0.0.13
      --worker-group w1.internal,w2.internal
      (repeat the flag to test multiple worker counts)
    """
    if not arg:
        raise argparse.ArgumentTypeError(
            "At least one --worker-group must be supplied (comma-separated hostnames/IPs)."
        )

    groups: List[tuple[str, ...]] = []
    for item in arg:
        members = tuple(name.strip() for name in item.split(",") if name.strip())
        if not members:
            raise argparse.ArgumentTypeError(
                "Worker group definitions must include at least one hostname/IP."
            )
        # optional de-dup within a group while preserving order:
        seen = set()
        members = tuple(m for m in members if not (m in seen or seen.add(m)))
        groups.append(members)

    if not groups:
        raise argparse.ArgumentTypeError("At least one --worker-group must be supplied.")
    return groups



def parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Warmup duration must be non-negative")
    return parsed


def detect_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
    except subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip()


def run_pgbench(
        *,
        host: str,
        port: int,
        database: str,
        user: str,
        clients: int,
        threads: int,
        duration: int,
        rw_mix: str = "90/10",
) -> tuple[float, float, float, float]:
    """Execute pgbench once against a remote coordinator and return TPS plus latency metrics."""

    # Unique local log prefix for pgbench (-l --log-prefix <prefix>)
    log_prefix = f"/tmp/pgbench_log_{uuid.uuid4().hex}_"

    # Choose script path based on mix
    if rw_mix.strip() in ("90/10", "read", "readonly"):
        workload_file = "bench_scripts/90-10.sql"
    else:  # 50/50 or write-heavy
        workload_file = "bench_scripts/50-50.sql"

    # Build pgbench command to run LOCALLY on the client VM (no docker)
    cmd = [
        "pgbench",
        "-h", host,
        "-p", str(port),
        "-U", user,
        "-c", str(clients),
        "-j", str(threads),
        "-T", str(duration),
        "-P", "10",
        "-r",
        "-l",
        "--log-prefix", log_prefix,
        "-f", workload_file,
        database,
    ]

    # Run and capture stdout/stderr text
    output = stream_command(cmd)

    # Collect latency samples from local pgbench logs (microseconds → ms)
    latencies_ms: list[float] = []
    try:
        import glob, os
        for path in glob.glob(f"{log_prefix}*"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Heuristic: last numeric token is latency (µs) in pgbench log format
                        parts = line.split()
                        lat_us = None
                        for tok in reversed(parts):
                            try:
                                lat_us = float(tok)
                                break
                            except ValueError:
                                continue
                        if lat_us is not None:
                            latencies_ms.append(lat_us / 1000.0)
            finally:
                # Clean up each log file
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        # If log parsing fails, we’ll fall back to pgbench summary parsing below
        pass

    # Parse pgbench summary from stdout
    tps: float | None = None
    lat_mean_output: float | None = None
    lat_p95_output: float | None = None
    lat_p99_output: float | None = None

    tps_re = re.compile(r"^\s*tps\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    mean_re = re.compile(r"latency\s+average\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    # Allow old/new percentile formats (95th / 95% / about 95%)
    p95_re = re.compile(r"latency[^\n]*?(?:95)(?:th)?(?:\s*percentile|%)[^0-9]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    p99_re = re.compile(r"latency[^\n]*?(?:99)(?:th)?(?:\s*percentile|%)[^0-9]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    for line in output.splitlines():
        if (m := tps_re.search(line)):
            tps = float(m.group(1)); continue
        if (m := mean_re.search(line)):
            lat_mean_output = float(m.group(1)); continue
        if (m := p95_re.search(line)):
            lat_p95_output = float(m.group(1)); continue
        if (m := p99_re.search(line)):
            lat_p99_output = float(m.group(1)); continue

    if tps is None:
        raise CommandError("Failed to locate required pgbench metrics (tps) in output")

    # Prefer high-resolution per-transaction latencies from logs if available
    if latencies_ms:
        lat_mean = sum(latencies_ms) / len(latencies_ms)
        lat_p95 = percentile(latencies_ms, 0.95)
        lat_p99 = percentile(latencies_ms, 0.99)
    else:
        lat_mean = lat_mean_output
        lat_p95 = lat_p95_output if lat_p95_output is not None else float("nan")
        lat_p99 = lat_p99_output if lat_p99_output is not None else float("nan")

    if lat_mean is None:
        raise CommandError("Failed to locate required pgbench metrics (latency mean)")

    return tps, lat_mean, lat_p95, lat_p99



# Existing imports above

# Add this class near other helper definitions (before main())
class HostSampler(threading.Thread):
    def __init__(self, host: str, interval: float, ssh_binary: str, ssh_user: str | None, ssh_opts: str):
        super().__init__(daemon=True)
        self.host = host
        self.interval = interval
        self.ssh_binary = ssh_binary
        self.ssh_user = ssh_user
        self.ssh_opts = ssh_opts
        self._stop = threading.Event()
        self.cpu_samples: list[float] = []
        self.mem_samples: list[float] = []

    def run(self) -> None:
        try:
            prev = _read_proc_stat_line(self.host, self.ssh_binary, self.ssh_user, self.ssh_opts)
        except Exception:
            return
        while not self._stop.wait(self.interval):
            try:
                curr = _read_proc_stat_line(self.host, self.ssh_binary, self.ssh_user, self.ssh_opts)
                cpu_pct = _cpu_percent_from_two_reads(prev, curr)
                self.cpu_samples.append(cpu_pct)
                prev = curr
                mem_pct = _mem_percent(self.host, self.ssh_binary, self.ssh_user, self.ssh_opts)
                if not math.isnan(mem_pct):
                    self.mem_samples.append(mem_pct)
            except Exception:
                continue

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> tuple[float, float]:
        cpu_avg = statistics.fmean(self.cpu_samples) if self.cpu_samples else float("nan")
        mem_avg = statistics.fmean(self.mem_samples) if self.mem_samples else float("nan")
        return cpu_avg, mem_avg

    ###########

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pgbench from the client VM against a remote Citus coordinator.")
    parser.add_argument(
        "--design-csv",
        help="CSV with explicit run rows (sign table). Columns: run_id,workers,shards,scale,concurrency,rw_mix[,machine]",
    )
    parser.add_argument(
        "--clients",
        nargs="*",
        type=int,
        default=DEFAULT_CLIENTS,
        help="List of pgbench client counts to test (default: 1 2 4 8 12 16 24 32 48 64)",
    )
    parser.add_argument("--duration", type=int, default=60, help="Duration of each pgbench run in seconds (default: 60)")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=DEFAULT_SCALES,
        help="One or more pgbench scale factors to test (default: 10 100)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of pgbench worker threads (defaults to the matching client count)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV file for results (default: results/results_<timestamp>.csv)",
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Coordinator hostname or IP"
    )
    parser.add_argument(
        "--port",
        type=int, default=5432,
        help="Coordinator port (default: 5432)"
    )
    parser.add_argument(
        "--database",
        default="postgres",
        help="Database name to initialize and benchmark (default: postgres)",
    )
    parser.add_argument(
        "--user",
        default="postgres",
        help="Database user for pgbench and psql connections (default: postgres)",
    )
    parser.add_argument(
        "--worker-group",
        required=True,
        dest="worker_groups",
        action="append",
        help=(
            "Comma-separated list of worker hostnames/IPs for a single scenario. "
            "Repeat to evaluate multiple worker counts (e.g., '10.0.0.11,10.0.0.12')."
        ),
    )
    parser.add_argument(
        "--worker-port",
        type=int,
        default=5432,
        help="Port where worker Postgres instances listen (default: 5432)",
    )
    parser.add_argument(
        "--warmup",
        type=parse_non_negative_int,
        default=10,
        help=(
            "Seconds to pause between pgbench runs to allow the system to stabilize "
            "(default: 10)"
        ),
    )
    parser.add_argument(
        "--rw-mix",
        default="90/10",
        help="Read/write mix (90/10 or 50/50). Selects pgbench script to run."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Sample CPU and memory on coordinator and workers during each run"
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="SSH username for remote hosts (default: current user)"
    )
    parser.add_argument(
        "--ssh-opts",
        default="-o BatchMode=yes -o StrictHostKeyChecking=no",
        help="Extra ssh options (default: disable host key prompts)"
    )
    parser.add_argument(
        "--ssh-binary",
        default="ssh",
        help="SSH command to use (e.g., 'ssh' or 'gcloud compute ssh --ssh-flag=...')"
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # NEW: shared DB params for every psql/pgbench call
    DB = dict(
        host=args.host,      # <-- requires --host in parser
        port=args.port,      # <-- requires --port (default 5432) in parser
        database=args.database,
        user=args.user,
    )

    timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    # Output CSV path (unchanged)
    output_path = args.output
    if output_path is None:
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"results_{timestamp.strftime('%Y%m%d-%H%M%S')}.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Common bookkeeping
    results: List[dict[str, str]] = []
    git_commit = detect_git_commit()
    diagnostics_root = Path("results") / "diagnostics" / timestamp.strftime("%Y%m%d-%H%M%S")
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    # Design-csv mode?
    use_design = bool(getattr(args, "design_csv", None))

    # Always ensure pg_stat_statements once per session
    ensure_pg_stat_statements(**DB)

    if use_design:
        # === DESIGN CSV MODE: run the exact rows (e.g., your 16-run sign table) ===
        import csv as _csv

        with open(args.design_csv, "r", encoding="utf-8") as f:
            design_rows = list(_csv.DictReader(f))
        required_cols = {"run_id", "workers", "shards", "scale", "concurrency", "rw_mix"}
        missing = required_cols - {c.strip().lower() for c in design_rows[0].keys()}
        if missing:
            raise SystemExit(f"--design-csv is missing columns: {sorted(missing)}")

        for row in design_rows:
            run_id      = row["run_id"].strip()
            workers_raw = row["workers"].strip()
            # Allow either comma-separated or pipe-separated worker lists
            workers_raw = workers_raw.replace("|", ",")
            worker_group = tuple(w.strip() for w in workers_raw.split(",") if w.strip())

            shards      = int(row["shards"])
            scale       = int(row["scale"])
            concurrency = int(row["concurrency"])
            rw_mix      = row.get("rw_mix", "").strip() or "90/10"
            thread_count = args.threads or concurrency

            print(f"==> [{run_id}] Dropping existing pgbench tables")
            drop_pgbench_tables(**DB)

            print(f"==> [{run_id}] Configuring workers: {', '.join(worker_group)}")
            synchronize_workers(worker_group, worker_port=args.worker_port, **DB)

            print(f"==> [{run_id}] Creating empty pgbench schema")
            create_pgbench_schema(**DB)

            print(f"==> [{run_id}] Setting citus.shard_count={shards} and distributing tables")
            set_shard_count(**DB, shards=shards)
            distribute_pgbench_tables(**DB)

            print(f"==> [{run_id}] Loading pgbench data (scale={scale})")
            load_pgbench_data(scale=scale, **DB)

            if args.warmup:
                print(f"==> [{run_id}] Warm-up {args.warmup}s")
                time.sleep(args.warmup)

            print(f"==> [{run_id}] Running pgbench c={concurrency} j={thread_count} T={args.duration} mix={rw_mix}")

            # --- START monitoring (only if --monitor is set) ---
            hosts_to_monitor = []
            if getattr(args, "monitor", False):
                hosts_to_monitor.append(args.host)
                hosts_to_monitor.extend(worker_group)
                samplers = [
                    HostSampler(h, 1.0, args.ssh_binary, args.ssh_user, args.ssh_opts)
                    for h in hosts_to_monitor
                ]
                for s in samplers:
                    s.start()
            else:
                samplers = []

            # --- Run pgbench benchmark ---
            reset_pg_stat_statements(**DB)
            tps, lat_mean, lat_p95, lat_p99 = run_pgbench(
                clients=concurrency,
                threads=thread_count,
                duration=args.duration,
                rw_mix=rw_mix,
                **DB,
            )

            # --- STOP monitoring and summarize ---
            coord_cpu_avg = coord_mem_avg = worker_cpu_avg = worker_cpu_max = worker_mem_avg = worker_mem_max = float("nan")

            if samplers:
                for s in samplers:
                    s.stop(); s.join(timeout=2.0)

                if samplers:
                    coord_cpu_avg, coord_mem_avg = samplers[0].summary()
                    worker_cpu_avgs, worker_mem_avgs = [], []
                    for s in samplers[1:]:
                        c, m = s.summary()
                        worker_cpu_avgs.append(c)
                        worker_mem_avgs.append(m)
                    if worker_cpu_avgs:
                        worker_cpu_avg = statistics.fmean(worker_cpu_avgs)
                        worker_cpu_max = max(worker_cpu_avgs)
                    if worker_mem_avgs:
                        worker_mem_avg = statistics.fmean(worker_mem_avgs)
                        worker_mem_max = max(worker_mem_avgs)
                        # --- END monitoring section ---

            # Diagnostics folder per run
            run_dir = diagnostics_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # pg_stat_statements dump
            try:
                stats_rows = collect_pg_stat_statements(**DB)
            except CommandError as exc:
                (run_dir / "pg_stat_statements.error").write_text(str(exc), encoding="utf-8")
                stats_rows = []
            else:
                if stats_rows:
                    stats_path = run_dir / "pg_stat_statements.csv"
                    with stats_path.open("w", newline="") as stats_file:
                        writer = csv.DictWriter(stats_file, fieldnames=list(stats_rows[0].keys()))
                        writer.writeheader(); writer.writerows(stats_rows)

            # EXPLAIN capture (single representative query)
            query = select_query_for_explain(stats_rows) or DEFAULT_EXPLAIN_QUERY
            capture_explain_plan(query=query, destination=run_dir / "explain.txt", **DB)

            results.append({
                "timestamp_utc": timestamp.isoformat(),
                "git_commit": git_commit,
                "run_id": run_id,
                "scale": scale,
                "shards": shards,
                "concurrency": concurrency,
                "threads": thread_count,
                "workers": len(worker_group),
                "worker_group": ",".join(worker_group),
                "rw_mix": rw_mix,
                "tps": f"{tps:.6f}",
                "lat_mean_ms": f"{lat_mean:.6f}",
                "lat_p95_ms": f"{lat_p95:.6f}",
                "lat_p99_ms": f"{lat_p99:.6f}",
                "coord_cpu_avg_pct": f"{coord_cpu_avg:.2f}",
                "coord_mem_avg_pct": f"{coord_mem_avg:.2f}",
                "workers_cpu_avg_pct": f"{worker_cpu_avg:.2f}",
                "workers_cpu_max_pct": f"{worker_cpu_max:.2f}",
                "workers_mem_avg_pct": f"{worker_mem_avg:.2f}",
                "workers_mem_max_pct": f"{worker_mem_max:.2f}",
            })

    else:
        # === LEGACY LOOP MODE: scales × worker_groups × clients (what you already had) ===
        clients = args.clients
        worker_groups = parse_worker_groups(args.worker_groups)
        scales = args.scales

        min_clients = min(clients)
        max_clients = max(clients)
        explain_low_captured: set[tuple[int, tuple[str, ...]]] = set()
        explain_high_captured: set[tuple[int, tuple[str, ...]]] = set()

        for scale in scales:
            for worker_group in worker_groups:
                print("==> Dropping existing pgbench tables")
                drop_pgbench_tables(**DB)

                print(f"==> Configuring workers: {', '.join(worker_group)}")
                synchronize_workers(worker_group, worker_port=args.worker_port, **DB)

                print("==> Creating empty pgbench schema")
                create_pgbench_schema(**DB)

                # If you want per-scenario shards (e.g., from another flag), set here
                # set_shard_count(**DB, shards=args.shards)

                print("==> Distributing pgbench tables")
                distribute_pgbench_tables(**DB)

                print(f"==> Loading pgbench data (scale={scale})")
                load_pgbench_data(scale=scale, **DB)

                scenario_root = (
                        diagnostics_root
                        / f"scale_{scale}"
                        / f"workers_{len(worker_group)}_{'-'.join(worker_group)}"
                )
                scenario_root.mkdir(parents=True, exist_ok=True)

                for idx, client_count in enumerate(clients):
                    thread_count = args.threads or client_count
                    if idx > 0 and args.warmup:
                        print(f"==> Waiting {args.warmup}s before the next run")
                        time.sleep(args.warmup)

                    print(f"==> Running pgbench with {client_count} clients ({thread_count} threads) for {args.duration}s")
                    # --- START monitoring (only if --monitor is set) ---
                    hosts_to_monitor = []
                    if getattr(args, "monitor", False):
                        hosts_to_monitor.append(args.host)
                        hosts_to_monitor.extend(worker_group)
                        samplers = [
                            HostSampler(h, 1.0, args.ssh_binary, args.ssh_user, args.ssh_opts)
                            for h in hosts_to_monitor
                        ]
                        for s in samplers:
                            s.start()
                    else:
                        samplers = []
                    # --- END start monitoring section ---

                    reset_pg_stat_statements(**DB)
                    tps, lat_mean, lat_p95, lat_p99 = run_pgbench(
                        clients=client_count,
                        threads=thread_count,
                        duration=args.duration,
                        **DB,
                    )

                    # --- STOP monitoring and summarize ---
                    coord_cpu_avg = coord_mem_avg = worker_cpu_avg = worker_cpu_max = worker_mem_avg = worker_mem_max = float("nan")

                    if samplers:
                        for s in samplers:
                            s.stop()
                            s.join(timeout=2.0)

                        if samplers:
                            coord_cpu_avg, coord_mem_avg = samplers[0].summary()
                            worker_cpu_avgs, worker_mem_avgs = [], []
                            for s in samplers[1:]:
                                c, m = s.summary()
                                worker_cpu_avgs.append(c)
                                worker_mem_avgs.append(m)
                            if worker_cpu_avgs:
                                worker_cpu_avg = statistics.fmean(worker_cpu_avgs)
                                worker_cpu_max = max(worker_cpu_avgs)
                            if worker_mem_avgs:
                                worker_mem_avg = statistics.fmean(worker_mem_avgs)
                                worker_mem_max = max(worker_mem_avgs)
                    # --- END monitoring section ---

                    run_dir = scenario_root / f"clients_{client_count}_threads_{thread_count}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    stats_rows: List[dict[str, str]] = []
                    try:
                        stats_rows = collect_pg_stat_statements(**DB)
                    except CommandError as exc:
                        (run_dir / "pg_stat_statements.error").write_text(str(exc), encoding="utf-8")
                    else:
                        if stats_rows:
                            stats_path = run_dir / "pg_stat_statements.csv"
                            with stats_path.open("w", newline="") as stats_file:
                                writer = csv.DictWriter(stats_file, fieldnames=list(stats_rows[0].keys()))
                                writer.writeheader(); writer.writerows(stats_rows)

                    scenario_key = (scale, tuple(worker_group))
                    if client_count == min_clients and scenario_key not in explain_low_captured:
                        query = select_query_for_explain(stats_rows) or DEFAULT_EXPLAIN_QUERY
                        capture_explain_plan(query=query, destination=scenario_root / "explain_low.txt", **DB)
                        explain_low_captured.add(scenario_key)

                    if client_count == max_clients and scenario_key not in explain_high_captured:
                        query = select_query_for_explain(stats_rows) or DEFAULT_EXPLAIN_QUERY
                        capture_explain_plan(query=query, destination=scenario_root / "explain_high.txt", **DB)
                        explain_high_captured.add(scenario_key)

                    results.append({
                        "timestamp_utc": timestamp.isoformat(),
                        "git_commit": git_commit,
                        "scale": scale,
                        "duration": args.duration,
                        "clients": client_count,
                        "threads": thread_count,
                        "workers": len(worker_group),
                        "worker_group": ",".join(worker_group),
                        "tps": f"{tps:.6f}",
                        "lat_mean_ms": f"{lat_mean:.6f}",
                        "lat_p95_ms": f"{lat_p95:.6f}",
                        "lat_p99_ms": f"{lat_p99:.6f}",
                        "coord_cpu_avg_pct": f"{coord_cpu_avg:.2f}",
                        "coord_mem_avg_pct": f"{coord_mem_avg:.2f}",
                        "workers_cpu_avg_pct": f"{worker_cpu_avg:.2f}",
                        "workers_cpu_max_pct": f"{worker_cpu_max:.2f}",
                        "workers_mem_avg_pct": f"{worker_mem_avg:.2f}",
                        "workers_mem_max_pct": f"{worker_mem_max:.2f}",
                    })

    # Columns: add any new fields you captured in design mode
    fieldnames = [
        "timestamp_utc",
        "git_commit",
        # design fields (present for design mode; empty in legacy mode is fine)
        "run_id", "shards", "concurrency", "rw_mix",
        # legacy/common fields
        "scale", "duration", "clients", "threads",
        "workers", "worker_group",
        "tps", "lat_mean_ms", "lat_p95_ms", "lat_p99_ms",
        "coord_cpu_avg_pct", "coord_mem_avg_pct",
        "workers_cpu_avg_pct", "workers_cpu_max_pct",
        "workers_mem_avg_pct", "workers_mem_max_pct",
    ]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"==> Results written to {output_path}")


if __name__ == "__main__":
    main()
