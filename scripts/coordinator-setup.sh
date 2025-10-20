#!/usr/bin/env bash
set -euo pipefail

echo "=== Coordinator VM setup (PostgreSQL 15 + Citus + config) ==="

# ---- Inputs ----
read -rp "Allow connections from which CIDR (internal VPC range) [10.0.0.0/8]: " CIDR
CIDR="${CIDR:-10.0.0.0/8}"

read -rp "Database name to create/use [bench]: " DB_NAME
DB_NAME="${DB_NAME:-bench}"

read -rp "Database user [postgres]: " DB_USER
DB_USER="${DB_USER:-postgres}"

read -rsp "Set/confirm password for user '${DB_USER}': " DB_PASS
echo

PG_VER=15
CITUS_PKG="postgresql-${PG_VER}-citus-13.2"

# ---- Packages ----
export DEBIAN_FRONTEND=noninteractive
if ! command -v psql >/dev/null 2>&1; then
  curl -s https://install.citusdata.com/community/deb.sh | sudo bash
  sudo apt-get update -y
  sudo apt-get install -y "postgresql-${PG_VER}" "${CITUS_PKG}" postgresql-contrib
else
  echo "PostgreSQL already installed."
  sudo apt-get install -y "${CITUS_PKG}" postgresql-contrib || true
fi

# ---- Ensure a cluster exists and is running ----
if ! pg_lsclusters | grep -q "${PG_VER}\s\+main"; then
  sudo pg_createcluster "${PG_VER}" main --start
else
  sudo systemctl enable postgresql
  sudo systemctl start postgresql
fi

CONF_DIR="/etc/postgresql/${PG_VER}/main"
CONF="${CONF_DIR}/postgresql.conf"
HBA="${CONF_DIR}/pg_hba.conf"

# ---- postgresql.conf tuning ----
sudo sed -i "s/^#\?listen_addresses.*/listen_addresses = '*'/" "${CONF}"

# shared_preload_libraries: citus,pg_stat_statements
if grep -q "^shared_preload_libraries" "${CONF}"; then
  sudo sed -i "s|^shared_preload_libraries.*|shared_preload_libraries = 'citus,pg_stat_statements'|" "${CONF}"
else
  echo "shared_preload_libraries = 'citus,pg_stat_statements'" | sudo tee -a "${CONF}" >/dev/null
fi

# Other helpful settings
if ! grep -q "^max_connections" "${CONF}"; then
  echo "max_connections = 500" | sudo tee -a "${CONF}" >/dev/null
else
  sudo sed -i "s/^#\?max_connections.*/max_connections = 500/" "${CONF}"
fi
if ! grep -q "^track_io_timing" "${CONF}"; then
  echo "track_io_timing = on" | sudo tee -a "${CONF}" >/dev/null
fi

# ---- pg_hba.conf (allow client + workers) ----
ALLOW_LINE="host    all             all             ${CIDR}              md5"
grep -qF "$ALLOW_LINE" "${HBA}" || echo "$ALLOW_LINE" | sudo tee -a "${HBA}" >/dev/null

# ---- Restart to apply ----
sudo systemctl restart postgresql

# ---- Create/alter role, DB, and extensions ----
if sudo -u postgres psql -Atqc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
  sudo -u postgres psql -c "ALTER ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}' SUPERUSER;"
else
  sudo -u postgres psql -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}' SUPERUSER;"
fi

if ! sudo -u postgres psql -Atqc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
fi

sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS citus;"
sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

# ---- Final setup: ensure bench DB and restart ----
echo
echo "=== Ensuring 'bench' DB and Citus extension, restarting service ==="
sudo -u postgres createdb bench || true
sudo -u postgres psql -d bench -c "CREATE EXTENSION IF NOT EXISTS citus;"
sudo systemctl restart postgresql

echo
echo "=== Coordinator ready ==="
echo "Internal IP (for workers to allow):"
ip -br -4 addr show scope global | awk '{print $3}' | cut -d/ -f1
echo
echo "Remember: in your benchmark runs, use --host <COORDINATOR_INTERNAL_IP> and database '${DB_NAME}'."
