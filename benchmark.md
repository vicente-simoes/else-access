# Benchmark Guide — `run_benchmark.py`

This document explains **exactly** how to use `run_benchmark.py` to benchmark a **Citus** cluster from a **client VM**. It covers prerequisites, how the script works, every CLI argument, run modes, example commands, outputs, and troubleshooting.

> **TL;DR:** You run this script **on the client VM**. It connects to your **Citus coordinator**, configures the worker set, prepares the `pgbench` dataset, runs timed workloads (90/10 or 50/50), collects stats/explain plans, and writes a results CSV plus diagnostics.

---

## 0) What this script does (high level)

For each run, the script:
1. **Connects** to your Citus **coordinator** (`--host`, `--port`, `--database`, `--user`).
2. **Optionally reconfigures workers** to match the requested worker group(s).
3. **Creates an empty `pgbench` schema** (tables + PKs, no data).
4. Sets **`citus.shard_count`** (design mode) and **distributes** the tables.
5. **Loads data** at the requested **scale**.
6. Runs **`pgbench`** for the requested **duration** with **concurrency** and **workload mix** (90/10 read-heavy or 50/50).
7. **Collects `pg_stat_statements`**, grabs an **EXPLAIN (ANALYZE, BUFFERS)** for a representative query, and
8. **Appends** a row to a **results CSV** and writes **diagnostics** to per-run folders.

All orchestration is done **from the client VM** using `psql` and `pgbench` CLI tools. No Docker is required in Stage 2.

---

## 1) Prerequisites

### Client VM (driver)
Install Postgres client tools (provides `psql` and `pgbench`) and set auth:
```bash
sudo apt-get update
sudo apt-get install -y postgresql-client postgresql-contrib
export PGPASSWORD='<postgres_password>'   # used by psql/pgbench with -U
```
Clone your repo onto the client VM so it can run `run_benchmark.py` and find the workload scripts in `bench_scripts/`.

### Coordinator & Worker VMs
- Install PostgreSQL (version you use) and **Citus** on **all** DB VMs.
- In `postgresql.conf`:
  - `shared_preload_libraries = 'citus,pg_stat_statements'`
  - `listen_addresses = '*'`
  - `max_connections = 500` (or higher if needed)
  - (optional) `track_io_timing = on`
- In `pg_hba.conf`:
  - Allow **client VM** to connect to **coordinator** (md5).
  - Allow **coordinator** to connect to **workers** (md5).
- Open port **5432** in your VPC so the client can reach the coordinator and the coordinator can reach workers.
- Create the user/database you’ll use:
  ```bash
  sudo -u postgres psql -c "CREATE ROLE postgres WITH LOGIN PASSWORD '<postgres_password>' SUPERUSER;"
  sudo -u postgres createdb -O postgres bench
  ```

### Workload scripts (required files)
Place these in your repo at `bench_scripts/` (as referenced by the script):
- `bench_scripts/90-10.sql` — read-heavy (roughly 90% reads)
- `bench_scripts/50-50.sql` — balanced (≈ 50% reads / 50% writes)

Example contents:
```sql
-- bench_scripts/90-10.sql
\set aid random(1, 1000000)
SELECT abalance FROM pgbench_accounts WHERE aid = :aid;
```
```sql
-- bench_scripts/50-50.sql
\set aid1 random(1, 1000000)
\set aid2 random(1, 1000000)
BEGIN;
UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = :aid1;
SELECT abalance FROM pgbench_accounts WHERE aid = :aid2;
COMMIT;
```

---

## 2) Two run modes

The script supports **two** ways to specify experiments.

### A) **Design CSV mode** (exact rows; recommended for DOE/sign table)
Provide a CSV file with **explicit rows** to run in sequence:
```text
run_id,workers,shards,scale,concurrency,rw_mix
run01,10.0.0.11,32,10,32,90/10
run02,10.0.0.11,128,100,256,50/50
...
```
- `workers`: **comma-separated** list of worker hostnames/IPs for **that run**.
- `shards`: number used for `citus.shard_count` **before** distribution.
- `scale`: `pgbench` scale (dataset size).
- `concurrency`: `pgbench -c <clients>` for that run.
- `rw_mix`: `90/10` → uses `bench_scripts/90-10.sql`; `50/50` → uses `bench_scripts/50-50.sql`.

**Command:**
```bash
python run_benchmark.py \
  --host <COORD_IP> --port 5432 \
  --database bench --user postgres \
  --design-csv sign16.csv \
  --warmup 600 --duration 900
```
> In design mode, the script will set shard count and distribute **per row**, then load data for that scale and run `pgbench` once for that row.

### B) **Legacy loop mode** (Cartesian loops by scale × worker_groups × clients)
You pass lists and the script iterates their **cross product**:
```bash
python run_benchmark.py \
  --host <COORD_IP> --port 5432 \
  --database bench --user postgres \
  --scales 10 100 \
  --worker-group 10.0.0.11,10.0.0.12,10.0.0.13 \
  --worker-group 10.0.0.11,10.0.0.12,10.0.0.13,10.0.0.14,10.0.0.15 \
  --clients 32 256 512 \
  --rw-mix 90/10 \
  --warmup 10 --duration 120
```
For each **scale** and **worker group**, it will:
1. Drop tables → create empty schema → distribute → load data
2. Run `pgbench` once per **clients** value (threads default to clients if `--threads` not set).
> In legacy mode, shard count is **not** changed per scenario unless you add a custom flag and call `set_shard_count()` yourself (already implemented but commented in code).

---

## 3) All CLI arguments (exhaustive)

### Required database connection
- `--host <IP|DNS>` (required): Citus **coordinator** address.
- `--port <int>`: Coordinator port. **Default:** `5432`.
- `--database <name>`: Target DB (schema/data will be created here). **Default:** `postgres`.
- `--user <name>`: DB user for `psql`/`pgbench`. **Default:** `postgres`.

> The script uses environment variable `PGPASSWORD` for password auth. Set it before running.

### Worker configuration
- `--worker-group "<w1>,<w2>,..."` (repeatable, **required** if not using `--design-csv`):  
  A **comma-separated** list of worker hostnames/IPs. Repeat this flag to test different worker counts (e.g., 3 workers vs 9).
- `--worker-port <int>`: Worker Postgres port. **Default:** `5432`.

For each scenario (or design row), the script calls:
- `master_remove_node(...)` for workers not in the list;
- `master_add_node(...)` for missing workers.

### Design CSV mode
- `--design-csv <path>`: Provide a CSV with **exact runs**. Columns (case-insensitive):  
  `run_id, workers, shards, scale, concurrency, rw_mix[, machine]`  
  - `workers`: comma-separated list or use `|` in file (the script accepts either; `|` is turned into `,`).  
  - `shards`: integer used for `SET citus.shard_count = <shards>` before distributing.  
  - `scale`: integer `pgbench` scale.  
  - `concurrency`: used for `pgbench -c`.  
  - `rw_mix`: `"90/10"` or `"50/50"`. Selects the proper SQL script.  
  - `machine`: optional, carried into your own analysis (not used by the script).

If this flag is present, **legacy loops are ignored** and only the CSV rows run.

### Legacy loop mode (if `--design-csv` is not provided)
- `--scales <int> [<int> ...]`: List of `pgbench` scales to test. **Default:** `10 100`.
- `--clients <int> [<int> ...]`: List of client concurrencies. **Default:** `1 2 4 8 12 16 24 32 48 64`.
- `--rw-mix <str>`: `"90/10"` or `"50/50"` **for all runs** in the loop.  
  You can quickly try both by running the loop twice with different `--rw-mix` values.

### Timing & threads
- `--duration <int>`: Seconds to run `pgbench` per run. **Default:** `60`.
- `--warmup <int>`: Seconds to wait between runs. **Default:** `10`.
- `--threads <int>`: `pgbench -j` worker threads. **Default:** **matches clients** (if omitted).  
  > **Recommendation:** For DOE consistency, leave `--threads` unset so `threads = clients`.

### Output
- `--output <path>`: Destination CSV for results. Default is `results/results_<timestamp>.csv` (folder created automatically).  
  Diagnostics (EXPLAIN and `pg_stat_statements`) go under `results/diagnostics/<timestamp>/...`.

---

## 4) What happens in each run (step-by-step)

**Design CSV mode (per row):**
1. Drop existing `pgbench_*` tables (if any).
2. Sync worker membership to the given **worker list**.
3. Create **empty** `pgbench` schema: `pgbench -i -I dtp` (drop, create tables, create PKs).
4. `SET citus.shard_count = <shards>` (**per row**).
5. Distribute the tables (Citus: `create_distributed_table` / `create_reference_table`).
6. Load data: `pgbench -i -I gv -n -s <scale>` (generate data + vacuum only).
7. Optional warm-up sleep.
8. Run `pgbench`:
   - `-c <concurrency>`
   - `-j <threads>` (defaults to concurrency if `--threads` not given)
   - `-T <duration>` seconds
   - `-f bench_scripts/90-10.sql` **or** `bench_scripts/50-50.sql` based on `rw_mix`
9. Collect metrics:
   - `pg_stat_statements` (top entries, CSV)
   - `EXPLAIN (ANALYZE, BUFFERS)` (representative query from stats or a default)
10. Append one row to the results CSV.

**Legacy loop mode:** same structure, but shard count is not dynamically set per scenario (unless you add that call), and the workload mix is fixed by `--rw-mix` for the whole loop.

---

## 5) Results & diagnostics

### Results CSV columns
The script writes rows with the following fields (some are used only in design mode):
- `timestamp_utc` — ISO timestamp of run start.
- `git_commit` — short hash of your repo (or `unknown`).
- `run_id` — from design CSV (empty in legacy mode).
- `shards` — from design CSV (empty in legacy mode unless you add it).
- `concurrency` — from design CSV (legacy uses `clients` instead).
- `rw_mix` — `90/10` or `50/50` (`--rw-mix` in legacy mode).
- `scale` — dataset scale.
- `duration` — seconds per run.
- `clients` — legacy mode only (design mode uses `concurrency`).
- `threads` — thread count used (`--threads` or equals clients).
- `workers` — number of workers in the scenario.
- `worker_group` — comma-separated workers.
- `tps` — transactions per second (float).
- `lat_mean_ms`, `lat_p95_ms`, `lat_p99_ms` — latency stats (milliseconds).

### Diagnostics folders
For each run the script writes into `results/diagnostics/<timestamp>/<run_id or scenario>/`:
- `pg_stat_statements.csv` — top statements.
- `explain.txt` — result of `EXPLAIN (ANALYZE, BUFFERS)` for a representative query.
- (Console output contains the raw `pgbench` progress logs; per-transaction latencies are also parsed when available from `pgbench -l` logs).

---

## 6) Example workflows

### A) Run a 16-row fractional factorial design (sign table)
```bash
export PGPASSWORD='<postgres_password>'

# Small machine cluster (e.g., e2-standard-4 VMs)
python run_benchmark.py \
  --host 10.0.0.5 --port 5432 --database bench --user postgres \
  --design-csv sign8_clusterA.csv \
  --warmup 600 --duration 900

# Large machine cluster (e.g., n2-standard-8 VMs)
python run_benchmark.py \
  --host 10.0.0.25 --port 5432 --database bench --user postgres \
  --design-csv sign8_clusterB.csv \
  --warmup 600 --duration 900
```
Then **merge** the two result CSVs into a single 16-row table and add a `machine` column (`e2-standard-4` vs `n2-standard-8`).

### B) Worker sweep (scalability curve)
Create a design CSV with workers = {1,3,6,9} (repeat each 3× for variance), fixed shards/scale/mix; run with `--design-csv`.

### C) Legacy grid (quick ad-hoc tests)
```bash
python run_benchmark.py \
  --host 10.0.0.5 --port 5432 --database bench --user postgres \
  --scales 10 100 \
  --worker-group 10.0.0.11,10.0.0.12,10.0.0.13 \
  --worker-group 10.0.0.11,10.0.0.12,10.0.0.13,10.0.0.14,10.0.0.15,10.0.0.16 \
  --clients 32 128 256 512 \
  --rw-mix 50/50 \
  --warmup 30 --duration 300
```

---

## 7) Troubleshooting

**Q: `Command failed with exit code ...`**  
- The script raises `CommandError` when a subprocess (psql/pgbench) fails. The full command and its output are printed. Common causes:
  - `PGPASSWORD` not set or wrong password.
  - Firewall / `pg_hba.conf` does not allow the client/coordinator/worker connections.
  - Citus not loaded (`shared_preload_libraries` missing → restart Postgres).
  - Worker hostnames/IPs incorrect or not reachable from the coordinator.
  - `bench_scripts/90-10.sql` or `bench_scripts/50-50.sql` missing or path wrong.

**Q: Distribution failed (“cannot distribute non-empty table”).**  
- Ensure the order is: **create empty schema** → **set shard count** → **distribute** → **load data**. The script already does this in Stage 2.

**Q: TPS too low even at small concurrencies.**  
- Check whether the **client VM** is CPU-starved. Consider trying `--threads <vCPU count>` for a quick driver bottleneck check (outside DOE).

**Q: Workers aren’t added/removed as expected.**  
- The script calls `master_get_active_worker_nodes()` and then `master_add_node` / `master_remove_node`. Check coordinator logs and network rules.

**Q: Where did files go?**  
- Results: `results/results_<timestamp>.csv` (or your `--output`).  
- Diagnostics: `results/diagnostics/<timestamp>/...`

---

## 8) Reproducibility tips

- **Block by machine type** (run 8 DOE rows on Cluster A, 8 on Cluster B) and **randomize order within each block**.
- Keep **Postgres/Citus versions** fixed; note them in your report.
- Use the **same zone** and **disk type** across clusters.
- Store all design CSVs and raw results under `results/` and commit a summary with your report.

---

## 9) Quick reference (arguments)

| Flag | Required | Default | Description |
|---|---|---|---|
| `--host` | ✅ | — | Coordinator hostname/IP |
| `--port` |  | `5432` | Coordinator port |
| `--database` |  | `postgres` | Target database name |
| `--user` |  | `postgres` | Database user |
| `--worker-group` | ✅\* | — | One worker group per flag (comma-separated). Repeat for multiple groups. Required unless using `--design-csv`. |
| `--worker-port` |  | `5432` | Worker Postgres port |
| `--design-csv` |  | — | Run explicit rows: `run_id,workers,shards,scale,concurrency,rw_mix[,machine]` |
| `--scales` |  | `10 100` | (Legacy mode) pgbench scales to test |
| `--clients` |  | `1 2 4 8 12 16 24 32 48 64` | (Legacy mode) client concurrencies |
| `--rw-mix` |  | `90/10` | Workload mix: selects `bench_scripts/90-10.sql` or `50-50.sql` |
| `--threads` |  | `None` → equals `clients` | pgbench worker threads (`-j`) |
| `--duration` |  | `60` | Seconds per pgbench run |
| `--warmup` |  | `10` | Pause in seconds between runs |
| `--output` |  | `results/results_<ts>.csv` | Output CSV path |

---

## 10) Final sanity check (quick smoke test)

```bash
export PGPASSWORD='<postgres_password>'
python run_benchmark.py \
  --host 10.0.0.5 --port 5432 --database bench --user postgres \
  --worker-group 10.0.0.11,10.0.0.12,10.0.0.13 \
  --scales 10 --clients 16 32 \
  --rw-mix 90/10 \
  --warmup 5 --duration 30
```
You should see workers configured, schema created, tables distributed, data loaded, pgbench output streaming, and finally a **results CSV** plus **diagnostics** in `results/`.

---

*Happy benchmarking!*

