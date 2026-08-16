#!/usr/bin/env bash
# Idempotently create the SemRAG PostgreSQL role + database + pgvector
# extension. Run on the OFFLINE machine as a user that can reach postgres
# as a superuser.
#
#   scripts/setup_postgres.sh
#
# Auth fallbacks tried in order (first that works wins):
#   1. sudo -u postgres psql            (Debian/Ubuntu sudo layout)
#   2. psql -U postgres with PGPASSWORD (operator supplies the password)
#   3. psql as the current OS user (peer-auth, if that role is a superuser)
#   4. root user (Docker/containers)
#
# Safe to run repeatedly: existing role/db/extension are left untouched.
set -euo pipefail

DB_USER="${SEMRAG_PG_USER:-semrag}"
DB_PASS="${SEMRAG_PG_PASSWORD:-semrag}"
DB_NAME="${SEMRAG_PG_DATABASE:-semrag}"
DB_HOST="${SEMRAG_PG_HOST:-localhost}"
DB_PORT="${SEMRAG_PG_PORT:-5432}"

run_sql() {
    # Runs "$@" as a psql command under the first working superuser path.
    local sql="$1"
    if command -v sudo >/dev/null 2>&1 && sudo -n -u postgres true 2>/dev/null; then
        sudo -u postgres psql -v ON_ERROR_STOP=1 -c "$sql"
    elif [ -n "${PGPASSWORD:-}" ] && psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "SELECT 1" >/dev/null 2>&1; then
        psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -v ON_ERROR_STOP=1 -c "$sql"
    elif [ "$(id -u)" = "0" ] && su postgres -c "psql -c 'SELECT 1'" >/dev/null 2>&1; then
        su postgres -c "psql -v ON_ERROR_STOP=1 -c \"$sql\""
    elif psql -h "$DB_HOST" -p "$DB_PORT" -c "SELECT 1" >/dev/null 2>&1; then
        psql -h "$DB_HOST" -p "$DB_PORT" -v ON_ERROR_STOP=1 -c "$sql"
    else
        echo "error: could not reach PostgreSQL as a superuser." >&2
        echo "Try one of:" >&2
        echo "  sudo -u postgres psql ..." >&2
        echo "  PGPASSWORD=<postgres-pass> scripts/setup_postgres.sh" >&2
        echo "  (peer auth for the current OS user mapped to a superuser role)" >&2
        exit 1
    fi
}

if ! command -v psql >/dev/null 2>&1; then
    echo "error: 'psql' not found (install postgresql-client)" >&2
    exit 1
fi

echo "==> checking PostgreSQL reachability"
run_sql "SELECT version();" >/dev/null

echo "==> ensuring role $DB_USER"
run_sql "DO \$\$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
        CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS';
    ELSE
        ALTER ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';
    END IF;
END \$\$;"

echo "==> ensuring database $DB_NAME owned by $DB_USER"
if ! run_sql "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
    run_sql "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
fi

echo "==> ensuring pgvector extension"
run_sql "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null || {
    echo "error: pgvector extension could not be created." >&2
    echo "Install pgvector on the PostgreSQL server (apt install postgresql-16-pgvector) and retry." >&2
    exit 1
}

echo
echo "PostgreSQL ready: role=$DB_USER db=$DB_NAME host=$DB_HOST:$DB_PORT (pgvector available)."
echo "Point SEMRAG_DATABASE_URL at it, e.g.:"
echo "    postgresql+psycopg2://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME"