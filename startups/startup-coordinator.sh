#!/usr/bin/env bash
set -euo pipefail

# ======= CONFIGURE THESE =======
PG_VER=15
DB_NAME="bench"
DB_USER="postgres"
DB_PASS="CHANGE_ME_STRONG"
CLIENT_CIDR="10.0.0.0/16"   # client VM / management IP range
WORKERS_CIDR="10.0.0.0/16"  # worker nodes' IP range
LISTEN_ADDR="*"
MAX_CONN="500"
# ===============================

export DEBIAN_FRONTEND=noninteractive

# 1) Repos + packages (Postgres 15 + Citus + pg_stat_statements)
if ! command -v psql >/dev/null 2>&1; then
  curl -s https://install.citusdata.com/community/deb.sh | sudo bash
  sudo apt-get update -y
  sudo apt-get install -y "postgresql-${PG_VER}" "postgresql-${PG_VER}-citus" "postgresql-${PG_VER}-pg-stat-statements"
fi

CONF_DIR="/etc/postgresql/${PG_VER}/main"
CONF="${CONF_DIR}/postgresql.conf"
HBA="${CONF_DIR}/pg_hba.conf"

# 2) Basic postgres.conf tuning
sudo sed -i "s/^#\?listen_addresses.*/listen_addresses = '${LISTEN_ADDR}'/" "${CONF}"
sudo sed -i "s/^#\?max_connections.*/max_connections = ${MAX_CONN}/" "${CONF}"

# Ensure preload libraries line contains both citus and pg_stat_statements
if ! grep -q "shared_preload_libraries" "${CONF}"; then
  echo "shared_preload_libraries = 'citus,pg_stat_statements'" | sudo tee -a "${CONF}" >/dev/null
else
  sudo sed -i "s|^shared_preload_libraries.*|shared_preload_libraries = 'citus,pg_stat_statements'|" "${CONF}"
fi

# optional but useful
if ! grep -q "^track_io_timing" "${CONF}"; then
  echo "track_io_timing = on" | sudo tee -a "${CONF}" >/dev/null
fi

# 3) pg_hba rules (allow client VM and worker VMs)
# Avoid duplicating lines
for RULE in \
  "host    all             all             ${CLIENT_CIDR}           md5" \
  "host    all             all             ${WORKERS_CIDR}          md5"
do
  grep -qF "$RULE" "${HBA}" || echo "$RULE" | sudo tee -a "${HBA}" >/dev/null
done

# 4) Restart Postgres
sudo systemctl enable postgresql
sudo systemctl restart postgresql

# 5) Create role/db and extensions (idempotent)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}' SUPERUSER;"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"

# Enable extensions in the benchmark DB
sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS citus;"
sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

echo "Coordinator setup complete."
