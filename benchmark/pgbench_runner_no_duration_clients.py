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
import glob
import os
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
    result = subprocess.run(
        cmd,
        input=input_text,
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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    output_lines: List[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise CommandError(f"Command {' '.join(cmd)} failed with exit code {return_code}")
    return "".join(output_lines)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
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


def collect_pg_stat_statements(*, host: str, port: int, database: str, user: str, limit: int = 10) -> List[dict[str, str]]:
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
    for row in rows:
        candidate = row.get("query", "")
        normalized = _normalize_query_for_explain(candidate)
        if normalized:
            return normalized
    return None


def capture_explain_plan(*, host: str, port: int, database: str, user: str, query: str, destination: Path) -> None:
    sql = "\n".join([
        "SET client_min_messages TO WARNING;",
        "BEGIN;",
        f"EXPLAIN (ANALYZE, BUFFERS) {query};",
        "ROLLBACK;",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_command(build_psql_command(host=host, port=port, database=database, user=user), input_text=sql, check=False)
        output = result.stdout or ""
        if result.returncode != 0:
            output = f"Failed to capture plan (exit code {result.returncode}):\n{output}"
        destination.write_text(output, encoding="utf-8")
    except CommandError as exc:
        destination.write_text(str(exc), encoding="utf-8")


def run_pgbench(*, host: str, port: int, database: str, user: str, clients: int, threads: int, duration: int, rw_mix: str = "90/10") -> tuple[float, float, float, float]:
    """Run pgbench and return TPS, mean, 95th, and 99th latency (ms)."""

    log_prefix = f"/tmp/pgbench_log_{uuid.uuid4().hex}_"

    workload_file = "bench_scripts/90-10.sql" if rw_mix.strip() in ("90/10", "read", "readonly") else "bench_scripts/50-50.sql"

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

    output = stream_command(cmd)

    # Parse summary TPS/mean
    tps_re = re.compile(r"^\s*tps\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    mean_re = re.compile(r"latency\s+average\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    tps = lat_mean = None

    for line in output.splitlines():
        if (m := tps_re.search(line)):
            tps = float(m.group(1))
        elif (m := mean_re.search(line)):
            lat_mean = float(m.group(1))

    if tps is None or lat_mean is None:
        raise CommandError("Failed to extract TPS or mean latency from pgbench output.")

    # Parse per-transaction logs
    latencies_ms = []
    for path in glob.glob(f"{log_prefix}*"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        latencies_ms.append(float(parts[2]) / 1000.0)  # microseconds → ms
        except Exception:
            continue
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    lat_p95 = percentile(latencies_ms, 0.95) if latencies_ms else float("nan")
    lat_p99 = percentile(latencies_ms, 0.99) if latencies_ms else float("nan")

    return tps, lat_mean, lat_p95, lat_p99
