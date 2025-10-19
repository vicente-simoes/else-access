#!/usr/bin/env bash
set -euo pipefail

echo "=== Client VM setup (pgbench + Python + repo) ==="

# ---- Inputs ----
read -rp "Git repo URL (e.g. https://github.com/user/your-repo.git): " REPO_URL
REPO_DIR_DEFAULT="$(basename "${REPO_URL%.git}")"
read -rp "Local checkout directory [${REPO_DIR_DEFAULT}]: " REPO_DIR
REPO_DIR="${REPO_DIR:-$REPO_DIR_DEFAULT}"

read -rp "Database user [postgres]: " DB_USER
DB_USER="${DB_USER:-postgres}"

read -rsp "Database password for user '${DB_USER}' (used by psql/pgbench): " DB_PASS
echo

# ---- Packages ----
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip postgresql-client postgresql-contrib git

# ---- Repo ----
if [ -d "$REPO_DIR" ]; then
  echo "Repo directory '$REPO_DIR' already exists. Skipping clone."
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

# ---- Persist password for convenience (optional but handy) ----
PROFILE_FILE="$HOME/.bashrc"
if ! grep -q "export PGPASSWORD=" "$PROFILE_FILE"; then
  echo "export PGPASSWORD='${DB_PASS}'" >> "$PROFILE_FILE"
  echo "Added PGPASSWORD to $PROFILE_FILE (new shells will pick it up)."
fi
export PGPASSWORD="${DB_PASS}"

# ---- Minimal check ----
echo "pgbench version:"
pgbench --version || true
echo "Python version:"
python3 --version || true

echo
echo "=== Done. Next steps ==="
echo "1) cd '${REPO_DIR}'"
echo "2) Prepare or confirm workload scripts exist at bench_scripts/90-10.sql and bench_scripts/50-50.sql"
echo "   Example 90-10.sql:"
echo "     \\set aid int(random(1, 1000000))"
echo "     SELECT abalance FROM pgbench_accounts WHERE aid = :aid;"
echo "   Example 50-50.sql:"
echo "     \\set aid1 int(random(1, 1000000))"
echo "     \\set aid2 int(random(1, 1000000))"
echo "     BEGIN;"
echo "     UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = :aid1;"
echo "     SELECT abalance FROM pgbench_accounts WHERE aid = :aid2;"
echo "     COMMIT;"
echo
echo "You can now run your benchmark from this client VM."
