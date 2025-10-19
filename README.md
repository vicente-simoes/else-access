

### Set up a Python environment for the analytics workflow

Create an isolated virtual environment and install the dependencies used by `graph.py`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The virtual environment keeps the benchmarking tooling separate from any system-wide Python packages. Activate it (`source .venv/bin/activate`) before running the plotting scripts and deactivate it with `deactivate` when you are done.

## 1. Start the Citus cluster

1. Launch the containers in the background:

   ```bash
   docker compose up -d
   ```

   In addition to the coordinator and worker containers, Compose starts a short-lived `citus_bootstrap` service. The service
   waits for PostgreSQL to accept connections, registers every hostname listed in `CITUS_WORKERS` (defaults to the three bundled
   workers) with the coordinator via `master_add_node`, initializes the `pgbench` schema (default scale factor `10`), and
   distributes the tables so they are ready for the benchmark. Check its output
   with:

   ```bash
   docker compose logs -f citus_bootstrap
   ```

2. (Optional) Confirm that the workers were registered:

   ```bash
   docker compose exec citus_coordinator psql -U postgres -c "SELECT * FROM master_get_active_worker_nodes();"
   ```


## 2. Run the automated workload generator

The repository now includes `benchmark/run_benchmark.py`, a reproducible driver that prepares the schema and executes the Stage I workload end-to-end. From the repository root run:

```bash
python benchmark/run_benchmark.py --output results.csv
```

By default the script performs a full matrix of scalability experiments:

1. **Worker count sweep** – Reconfigures the coordinator to run with 1, 2, and 3 workers (matching the `docker-compose.yml` services) before each batch of tests.
2. **Dataset size sweep** – Re-initializes the `pgbench` schema at scale factors 10 and 100 so you can observe both memory-resident and memory-pressure behaviour.
3. **Concurrency sweep** – Executes `pgbench` at ten client counts (`1 2 4 8 12 16 24 32 48 64`) with matching thread counts, streaming the TPS output to the console.
4. **Result capture** – Writes a CSV row for every run that includes the timestamp, Git commit, scale factor, worker configuration, runtime duration, client/thread counts, measured TPS, and latency percentiles (`lat_mean_ms`, `lat_p95_ms`, `lat_p99_ms`).
5. **Bottleneck diagnostics** – Stores pg_stat_statements snapshots, low/high load EXPLAIN plans, and a docker stats snapshot for each run under `results/diagnostics/<timestamp>/`.


The default output path is `results/results_<timestamp>.csv`; providing `--output results.csv` overwrites the sample data shipped with the repository so the analytics workflow can consume your fresh run immediately. All major parameters are configurable:

Run `python benchmark/run_benchmark.py --help` to see the full argument list. The most commonly tuned options are summarized below:

| Option | Description | Default |
| --- | --- | --- |
| `--clients N [N ...]` | Sequence of client counts to test. The script uses the same values for the pgbench thread count unless `--threads` is supplied. | `1 2 4 8 12 16 24 32 48 64` |
| `--duration SECONDS` | Runtime per pgbench invocation. Increase this if you want longer steady-state windows or to compensate for noisy hosts. | `60` |
| `--scales N [N ...]` | Pgbench scale factors to initialize before each sweep. Multiple values are allowed; the database is reinitialized for every scale. | `10 100` |
| `--threads N` | Explicit thread count to pass to pgbench. If omitted, pgbench threads match the active client count. | _same as `--clients`_ |
| `--output PATH` | File to receive the consolidated CSV output. Parent directories are created automatically. | `results/results_<timestamp>.csv` |
| `--service NAME` | Docker Compose service that hosts the Citus coordinator. Change this if you rename the container in `docker-compose.yml`. | `citus_coordinator` |
| `--database NAME` | Database to initialize, distribute, and benchmark. | `postgres` |
| `--user NAME` | Role used for all psql and pgbench connections. | `postgres` |
| `--worker-group LIST` | Specify one or more worker scenarios. Each use of the flag defines a comma-separated list of worker services (for example, `--worker-group citus_worker1,citus_worker2`). Repeat the option to test several worker counts. | One-, two-, and three-worker groups derived from `citus_worker1..3` |
| `--worker-port PORT` | Port number where the worker containers accept connections. Adjust when running on a custom Compose topology. | `5432` |
| `--warmup SECONDS` | Pause this many seconds between pgbench runs to let the system settle. Set to `0` to disable the wait. | `10` |


## 4. Visualize the benchmark run with `graph.py`


The plotting utility can now generate an entire suite of scalability charts from a single benchmark run. By default it reads the consolidated CSV (`results.csv`) and emits one image per scenario under `plots/<run-id>/`, creating a fresh subdirectory for every timestamped run.


```bash
python graph.py --csv results.csv
```


### Choosing which plots to render


Use `--plots` to list one or more visualizations or pass `all` to render everything. The available plot keys are:

| Plot key | Description | Required column(s) |
| --- | --- | --- |
| `usl_throughput` *(alias: `throughput`, `usl`)* | Fits the Universal Scalability Law curve and prints the estimated contention (`α`) and coherency (`β`) coefficients. | `tps` |
| `throughput_raw` | Plots the raw throughput points without fitting the USL model. | `tps` |
| `speedup` | Normalizes throughput against the smallest client count in the scenario to highlight parallel speedup. | `tps` |
| `efficiency` | Plots parallel efficiency (speedup ÷ client count). | `tps` |
| `latency_mean` | Shows the average transaction latency. | `lat_mean_ms` |
| `latency_p95` | Highlights the 95th percentile latency tail. | `lat_p95_ms` |
| `latency_p99` | Highlights the 99th percentile latency tail. | `lat_p99_ms` |

Example: regenerate the USL curve and the efficiency plot for each scenario captured in the CSV:

```bash
python graph.py --csv results.csv --plots usl_throughput efficiency
```

## Shutting everything down

When you are finished, stop and remove the containers:

```bash
docker compose down -v
```

This brings the environment back to a clean state while preserving the Stage I artifacts in the repository.
