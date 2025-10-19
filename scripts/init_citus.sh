#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

DB="${POSTGRES_DB:-postgres}"
USER="${POSTGRES_USER:-postgres}"
PASSWORD="${POSTGRES_PASSWORD:-postgres}"
HOST="${POSTGRES_HOST:-citus_coordinator}"
PORT="${POSTGRES_PORT:-5432}"
WORKER_PORT="${CITUS_WORKER_PORT:-5432}"
SCALE="${PGBENCH_SCALE:-10}"
IFS=' ' read -r -a WORKERS <<< "${CITUS_WORKERS:-citus_worker1 citus_worker2 citus_worker3}"

export PGPASSWORD="$PASSWORD"

log "Waiting for Postgres at ${HOST}:${PORT}/${DB} to become ready"
until pg_isready -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" >/dev/null 2>&1; do
  sleep 1
  log "Still waiting for Postgres..."
done

log "Ensuring Citus extension is available and registering workers"
register_sql="CREATE EXTENSION IF NOT EXISTS citus;"
for worker in "${WORKERS[@]}"; do
  register_sql+="
DO \$\$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM master_get_active_worker_nodes()
        WHERE node_name = '${worker}' AND node_port = ${WORKER_PORT}
    ) THEN
        PERFORM master_add_node('${worker}', ${WORKER_PORT});
    END IF;
END;
\$\$;"
done

psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -v ON_ERROR_STOP=1 <<<"$register_sql"

log "Checking for existing pgbench schema"
if [[ "$(psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -tAc "SELECT to_regclass('pgbench_accounts') IS NOT NULL")" != "t" ]]; then
  log "Initializing pgbench schema with scale ${SCALE}"
  pgbench -h "$HOST" -p "$PORT" -U "$USER" -i -I dtgvp -s "$SCALE" "$DB"
else
  log "pgbench schema already present, skipping initialization"
fi

log "Marking pgbench tables as distributed/reference where required"
psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS citus;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_dist_partition
        WHERE logicalrelid = 'pgbench_accounts'::regclass
    ) THEN
        PERFORM create_distributed_table('pgbench_accounts', 'aid');
    END IF;
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_dist_partition
        WHERE logicalrelid = 'pgbench_branches'::regclass
          AND partmethod = 'n'
    ) THEN
        PERFORM create_reference_table('pgbench_branches');
    END IF;
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_dist_partition
        WHERE logicalrelid = 'pgbench_tellers'::regclass
          AND partmethod = 'n'
    ) THEN
        PERFORM create_reference_table('pgbench_tellers');
    END IF;
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_dist_partition
        WHERE logicalrelid = 'pgbench_history'::regclass
    ) THEN
        PERFORM create_distributed_table('pgbench_history', 'tid');
    END IF;
END;
$$;
SQL

log "Citus bootstrap complete"
