#!/usr/bin/env bash
set -euo pipefail

echo "=== Worker VM setup (PostgreSQL 15 + Citus + allow coordinator) ==="

# ---- Inputs ----
read -rp "Coordinator INTERNAL IP (e.g., 10.132.0.5): " COORD_IP
read -rp "Auth mode from coordinator (trust/md5) [trust]: " AUTH_MODE
AUTH_MODE="${AUTH_MODE:-trust}"
if [[ "$AUTH_MODE" != "trust" && "$AUTH_MODE" != "md5" ]]; then
  echo "Invalid auth mode. Choose 'trust' or 'md5'." >&2
  exit 1
fi

read -rp "Database name to create/use [bench]: " DB_NAME
DB_NAME="${DB_NAME:-bench}"

read -rp "Database user [postgres]: " DB_USER
DB_USER="${DB_USER:-postgres}"

DB_PASS=""
if [[ "$AUTH_MODE" == "md5" ]]; then
  read -rsp "Set/confirm password for user '${DB_USER}' (coordinator will need this in its .pgpass): " DB_PASS
  echo
fi

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

if grep -q "^shared_preload_libraries" "${CONF}"; then
  sudo sed -i "s|^shared_preload_libraries.*|shared_preload_libraries = 'citus,pg_stat_statements'|" "${CONF}"
else
  echo "shared_preload_libraries = 'citus,pg_stat_statements'" | sudo tee -a "${CONF}" >/dev/null
fi

if ! grep -q "^max_connections" "${CONF}"; then
  echo "max_connections = 500" | sudo tee -a "${CONF}" >/dev/null
else
  sudo sed -i "s/^#\?max_connections.*/max_connections = 500/" "${CONF}"
fi

if ! grep -q "^track_io_timing" "${CONF}"; then
  echo "track_io_timing = on" | sudo tee -a "${CONF}" >/dev/null
fi

# ---- pg_hba: allow coordinator ----
ALLOW_LINE="host    all             all             ${COORD_IP}/32         ${AUTH_MODE}"
# Avoid duplicates:
grep -qF "$ALLOW_LINE" "${HBA}" || echo "$ALLOW_LINE" | sudo tee -a "${HBA}" >/dev/null

# ---- Restart to apply ----
sudo systemctl restart postgresql

# ---- Create/alter role, DB, and extensions ----
if sudo -u postgres psql -Atqc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
  if [[ "$AUTH_MODE" == "md5" ]]; then
    sudo -u postgres psql -c "ALTER ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}' SUPERUSER;"
  fi
else
  if [[ "$AUTH_MODE" == "md5" ]]; then
    sudo -u postgres psql -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}' SUPERUSER;"
  else
    # password not needed when using trust, but create user anyway:
    sudo -u postgres psql -c "CREATE ROLE ${DB_USER} WITH LOGIN SUPERUSER;"
  fi
fi

if ! sudo -u postgres psql -Atqc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
fi

sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS citus;"
sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

echo
echo "=== Worker ready ==="
echo "This worker now allows the coordinator ${COORD_IP} via '${AUTH_MODE}'."
if [[ "$AUTH_MODE" == "md5" ]]; then
  cat <<EOF

NOTE: Since you chose 'md5', you must add this line on the *coordinator* to ~postgres/.pgpass (and chmod 600):
${COORD_IP}:5432:${DB_NAME}:${DB_USER}:${DB_PASS}

Without that, master_add_node() will fail due to missing password.
EOF
fi
