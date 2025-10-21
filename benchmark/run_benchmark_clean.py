from __future__ import annotations

import argparse
import csv
import io
import math
import re
import subprocess
import sys
import time
import uuid
import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

# ---------- Defaults ----------
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_SCALES = [10, 100]
DEFAULT_EXPLAIN_QUERY = "SELECT abalance FROM pgbench_accounts WHERE aid = 1"

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- Errors ----------
class CommandError(RuntimeError):
    """Raised when a shell command exits with a non-zero status."""


# ---------- Shell helpers ----------
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


# ---------- Small utils ----------
def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    w = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * w


def build_psql_command(*, host: str, port: int, database: str, user: str, extra_args: Sequence[str] | None = None) -> List[str]:
    cmd = ["psql", "-v", "ON_ERROR_STOP=1", "-h", host, "-p", str(port), "-U", user, "-d", database]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


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
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


# ---------- DB helpers ----------
def ensure_pg_stat_statements(*, host: str, port: int, database: str, user: str) -> None:
    run_command(build_psql_command(host=host, port=port, database=database, user=user),
                input_text="CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")


def reset_pg_stat_statements(*, host: str, port: int, database: str, user: str) -> None:
    run_command(build_psql_command(host=host, port=port, database=database, user=user),
                input_text="SELECT pg_stat_statements_reset();")


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
    q = (query or "").strip()
    if not q:
        return None
    first = q.split(";")[0].strip()
    if not first:
        return None
    if not first.upper().startswith(("SELECT", "UPDATE", "INSERT", "DELETE")):
        return None
    return re.sub(r":[a-zA-Z_][a-zA-Z0-9_]*", "1", first)


def select_query_for_explain(rows: Sequence[dict[str, str]]) -> str | None:
    for row in rows:
        norm = _normalize_query_for_explain(row.get("query", ""))
        if norm:
            return norm
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
        out = result.stdout or ""
        if result.returncode != 0:
            out = f"Failed to capture plan (exit code {result.returncode}):\n{out}"
        destination.write_text(out, encoding="utf-8")
    except CommandError as exc:
        destination.write_text(str(exc), encoding="utf-8")


# ---------- Citus helpers ----------
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
    run_command(build_psql_command(host=host, port=port, database=database, user=user),
                input_text=f"SET citus.shard_count = {int(shards)};")


def synchronize_workers(
        workers: Iterable[str], *, host: str, port: int, database: str, user: str, worker_port: int
) -> None:
    desired = list(workers)
    desired_set = set(desired)
    active = fetch_active_workers(host=host, port=port, database=database, user=user)

    # Remove extras
    removal = [
        f"SELECT master_remove_node('{node_name}', {node_port});"
        for (node_name, node_port) in active if node_name not in desired_set
    ]
    if removal:
        run_command(build_psql_command(host=host, port=port, database=database, user=user),
                    input_text="\n".join(removal))

    # Add missing
    statements = ["CREATE EXTENSION IF NOT EXISTS citus;"]
    for w in desired:
        statements.append(
            "DO $$\nBEGIN\n"
            f"    IF NOT EXISTS (\n"
            f"        SELECT 1 FROM master_get_active_worker_nodes()\n"
            f"        WHERE node_name = '{w}' AND node_port = {worker_port}\n"
            f"    ) THEN\n"
            f"        PERFORM master_add_node('{w}', {worker_port});\n"
            f"    END IF;\n"
            "END;\n$$;"
        )
    run_command(build_psql_command(host=host, port=port, database=database, user=user),
                input_text="\n".join(statements))


# ---------- pgbench dataset ----------
def create_pgbench_schema(*, host: str, port: int, database: str, user: str) -> None:
    run_command(["pgbench", "-h", host, "-p", str(port), "-i", "-I", "dtp", "-U", user, database])


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
    run_command(["pgbench", "-h", host, "-p", str(port), "-i", "-I", "gv", "-n", "-s", str(scale), "-U", user, database])


def drop_pgbench_tables(*, host: str, port: int, database: str, user: str) -> None:
    sql = "\n".join([
        "DROP TABLE IF EXISTS pgbench_history CASCADE;",
        "DROP TABLE IF EXISTS pgbench_tellers CASCADE;",
        "DROP TABLE IF EXISTS pgbench_accounts CASCADE;",
        "DROP TABLE IF EXISTS pgbench_branches CASCADE;",
    ])
    run_command(build_psql_command(host=host, port=port, database=database, user=user), input_text=sql)


# ---------- pgbench runner ----------
def run_pgbench(
        *, host: str, port: int, database: str, user: str,
        concurrency: int, threads: int, duration: int, rw_mix: str = "90/10"
) -> tuple[float, float, float, float]:
    """Run pgbench and return TPS, mean, 95th, and 99th latency (ms)."""
    log_prefix = f"/tmp/pgbench_log_{uuid.uuid4().hex}_"
    workload_file = "bench_scripts/90-10.sql" if rw_mix.strip() in ("90/10", "read", "readonly") else "bench_scripts/50-50.sql"

    cmd = [
        "pgbench",
        "-h", host, "-p", str(port), "-U", user,
        "-c", str(concurrency),
        "-j", str(threads),
        "-T", str(duration),
        "-P", "10",
        "-r",
        "-l", "--log-prefix", log_prefix,
        "-f", workload_file,
        database,
    ]
    output = stream_command(cmd)

    # Summary TPS & mean
    tps_re = re.compile(r"^\s*tps\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    mean_re = re.compile(r"latency\s+average\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    tps = lat_mean = None
    for line in output.splitlines():
        m = tps_re.search(line)
        if m:
            tps = float(m.group(1))
            continue
        m = mean_re.search(line)
        if m:
            lat_mean = float(m.group(1))
    if tps is None or lat_mean is None:
        raise CommandError("Failed to extract TPS or mean latency from pgbench output.")

    # Per-transaction latencies from logs (μs → ms)
    lat_ms: List[float] = []
    for path in glob.glob(f"{log_prefix}*"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        lat_ms.append(float(parts[2]) / 1000.0)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    p95 = percentile(lat_ms, 0.95) if lat_ms else float("nan")
    p99 = percentile(lat_ms, 0.99) if lat_ms else float("nan")
    return tps, lat_mean, p95, p99


# ---------- CLI + Orchestration ----------
def parse_concurrency(arg: List[str] | None) -> List[int]:
    if not arg:
        return DEFAULT_CONCURRENCY
    return [int(x) for x in arg]


def parse_worker_groups(arg: List[str] | None) -> List[tuple[str, ...]]:
    if not arg:
        raise argparse.ArgumentTypeError("At least one --worker-group must be supplied (comma-separated hostnames/IPs).")
    groups: List[tuple[str, ...]] = []
    for item in arg:
        members = tuple(name.strip() for name in item.split(",") if name.strip())
        if not members:
            raise argparse.ArgumentTypeError("Worker group definitions must include at least one hostname/IP.")
        seen = set()
        members = tuple(m for m in members if not (m in seen or seen.add(m)))
        groups.append(members)
    return groups


def nonneg_int(value: str) -> int:
    try:
        v = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative")
    return v


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pgbench from the client VM against a remote Citus coordinator.")
    p.add_argument("--design-csv",
                   help="CSV with explicit rows: run_id,workers,shards,scale,concurrency,rw_mix[,machine]")
    p.add_argument("--concurrency", nargs="*", type=int, default=DEFAULT_CONCURRENCY,
                   help="List of pgbench concurrency values to test (default: common series)")
    p.add_argument("--duration", type=int, default=60, help="Duration of each pgbench run in seconds (default: 60)")
    p.add_argument("--scales", type=int, nargs="+", default=DEFAULT_SCALES,
                   help="One or more pgbench scale factors to test (default: 10 100)")
    p.add_argument("--threads", type=int, default=None,
                   help="Number of pgbench worker threads (defaults to concurrency)")
    p.add_argument("--output", type=Path, default=None,
                   help="Destination CSV file (default: results/results_<timestamp>.csv)")
    p.add_argument("--host", required=True, help="Coordinator hostname or IP")
    p.add_argument("--port", type=int, default=5432, help="Coordinator port (default: 5432)")
    p.add_argument("--database", default="postgres", help="Database name (default: postgres)")
    p.add_argument("--user", default="postgres", help="Database user (default: postgres)")
    p.add_argument("--worker-group", required=True, dest="worker_groups", action="append",
                   help="Comma-separated worker hostnames/IPs. Repeat flag to test multiple worker counts.")
    p.add_argument("--worker-port", type=int, default=5432, help="Worker Postgres port (default: 5432)")
    p.add_argument("--warmup", type=nonneg_int, default=10, help="Seconds to pause between runs (default: 10)")
    p.add_argument("--rw-mix", default="90/10", help="Read/write mix (90/10 or 50/50).")
    # NEW: shards list for LOOP mode
    p.add_argument("--shards", type=int, nargs="+",
                   help="Shard counts to test in loop mode (e.g., --shards 16 32 64).")
    # NEW: include git commit only if requested
    p.add_argument("--git_commit", action="store_true",
                   help="Include git commit column/value in results CSV.")
    return p


def prepare_dataset(*, DB, worker_group: tuple[str, ...], scale: int, worker_port: int, shards: int | None = None) -> None:
    print("==> Dropping existing pgbench tables")
    drop_pgbench_tables(**DB)
    print(f"==> Configuring workers: {', '.join(worker_group)}")
    synchronize_workers(worker_group, worker_port=worker_port, **DB)
    print("==> Creating empty pgbench schema")
    create_pgbench_schema(**DB)
    if shards is not None:
        print(f"==> Setting citus.shard_count={shards}")
        set_shard_count(**DB, shards=shards)
    print("==> Distributing pgbench tables")
    distribute_pgbench_tables(**DB)
    print(f"==> Loading pgbench data (scale={scale})")
    load_pgbench_data(scale=scale, **DB)


def run_and_record(*, DB, diagnostics_root: Path, results: List[dict[str, str]],
                   scale: int, worker_group: tuple[str, ...],
                   concurrency: int, duration: int, threads: int | None,
                   rw_mix: str, git_commit_value: str | None,
                   extra_cols: dict[str, str | int] | None = None,
                   explain_markers: dict[str, bool] | None = None) -> None:
    if explain_markers is None:
        explain_markers = {}
    th = threads or concurrency

    if extra_cols and extra_cols.get("run_id"):
        label_dir = diagnostics_root / str(extra_cols["run_id"])
    else:
        label_dir = (diagnostics_root / f"scale_{scale}" /
                     f"workers_{len(worker_group)}_{'-'.join(worker_group)}" /
                     f"concurrency_{concurrency}_threads_{th}")
    label_dir.mkdir(parents=True, exist_ok=True)

    reset_pg_stat_statements(**DB)

    print(f"==> Running pgbench concurrency={concurrency} threads={th} T={duration}s mix={rw_mix}")
    tps, lat_mean, lat_p95, lat_p99 = run_pgbench(
        concurrency=concurrency,
        threads=th,
        duration=duration,
        rw_mix=rw_mix,
        **DB,
    )

    # Collect pg_stat_statements (best-effort)
    try:
        stats_rows = collect_pg_stat_statements(**DB)
    except CommandError as exc:
        (label_dir / "pg_stat_statements.error").write_text(str(exc), encoding="utf-8")
        stats_rows = []
    else:
        if stats_rows:
            stats_path = label_dir / "pg_stat_statements.csv"
            with stats_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
                writer.writeheader(); writer.writerows(stats_rows)

    if stats_rows and (explain_markers.get("capture_explain") is True):
        query = select_query_for_explain(stats_rows) or DEFAULT_EXPLAIN_QUERY
        capture_explain_plan(query=query, destination=label_dir / "explain.txt", **DB)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scale": scale,
        "workers": len(worker_group),
        "worker_group": ",".join(worker_group),
        "concurrency": concurrency,
        "threads": th,
        "rw_mix": rw_mix,
        "tps": f"{tps:.6f}",
        "lat_mean_ms": f"{lat_mean:.6f}",
        "lat_p95_ms": f"{lat_p95:.6f}",
        "lat_p99_ms": f"{lat_p99:.6f}",
    }
    # Optional extras (design mode or shards in loop mode)
    if extra_cols:
        for k, v in extra_cols.items():
            if v is not None:
                row[k] = str(v)
    # Optional git commit
    if git_commit_value is not None:
        row["git_commit"] = git_commit_value

    results.append(row)


def main(argv: List[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    DB = dict(host=args.host, port=args.port, database=args.database, user=args.user)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    # Output path
    out_path = args.output or (Path("results") / f"results_{timestamp.strftime('%Y%m%d-%H%M%S')}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[dict[str, str]] = []
    git_commit_value = detect_git_commit() if args.git_commit else None  # only compute/include if requested
    diagnostics_root = Path("results") / "diagnostics" / timestamp.strftime("%Y%m%d-%H%M%S")
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    # Ensure pg_stat_statements
    ensure_pg_stat_statements(**DB)

    # DESIGN CSV MODE
    if args.design_csv:
        with open(args.design_csv, "r", encoding="utf-8") as f:
            design_rows = list(csv.DictReader(f))

        required = {"run_id", "workers", "shards", "scale", "concurrency", "rw_mix"}
        missing = required - {c.strip().lower() for c in design_rows[0].keys()}
        if missing:
            raise SystemExit(f"--design-csv is missing columns: {sorted(missing)}")

        for row in design_rows:
            run_id = row["run_id"].strip()
            workers_raw = row["workers"].replace("|", ",").strip()
            worker_group = tuple(w.strip() for w in workers_raw.split(",") if w.strip())
            shards = int(row["shards"])
            scale = int(row["scale"])
            concurrency = int(row["concurrency"])
            rw_mix = (row.get("rw_mix") or "90/10").strip()
            threads = args.threads or concurrency

            prepare_dataset(DB=DB, worker_group=worker_group, scale=scale, worker_port=args.worker_port, shards=shards)

            if args.warmup:
                print(f"==> Warm-up {args.warmup}s")
                time.sleep(args.warmup)

            run_and_record(
                DB=DB, diagnostics_root=diagnostics_root, results=results,
                scale=scale, worker_group=worker_group, concurrency=concurrency,
                duration=args.duration, threads=threads, rw_mix=rw_mix,
                git_commit_value=git_commit_value,
                extra_cols={"run_id": run_id, "shards": shards},
                explain_markers={"capture_explain": True},
            )

    # LOOP MODE (now supports --shards)
    else:
        concurrencies = parse_concurrency(args.concurrency)
        worker_groups = parse_worker_groups(args.worker_groups)
        scales = args.scales
        shards_list = args.shards if args.shards else [None]  # iterate once if not provided

        for scale in scales:
            for wg in worker_groups:
                for shards in shards_list:
                    prepare_dataset(DB=DB, worker_group=wg, scale=scale, worker_port=args.worker_port, shards=shards)

                    # capture explain only for min/max concurrency once per (scale, wg, shards)
                    low_c = min(concurrencies)
                    high_c = max(concurrencies)

                    for idx, c in enumerate(concurrencies):
                        th = args.threads or c
                        if idx > 0 and args.warmup:
                            print(f"==> Waiting {args.warmup}s before the next run")
                            time.sleep(args.warmup)

                        run_and_record(
                            DB=DB, diagnostics_root=diagnostics_root, results=results,
                            scale=scale, worker_group=wg, concurrency=c, duration=args.duration,
                            threads=th, rw_mix=args.rw_mix, git_commit_value=git_commit_value,
                            extra_cols=({"shards": shards} if shards is not None else None),
                            explain_markers={"capture_explain": c in (low_c, high_c)},
                        )

    # Build CSV headers dynamically
    fieldnames = [
        "timestamp_utc",
        # git_commit only if requested
        *(["git_commit"] if args.git_commit else []),
        # design-only columns (present when available)
        "run_id", "shards",
        # common columns
        "scale", "workers", "worker_group",
        "concurrency", "threads", "rw_mix",
        "tps", "lat_mean_ms", "lat_p95_ms", "lat_p99_ms",
    ]
    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"==> Results written to {out_path}")


if __name__ == "__main__":
    main()
