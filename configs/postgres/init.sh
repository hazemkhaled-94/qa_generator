#!/bin/bash
# Runs once on first boot of an empty postgres_data volume.
# Idempotent — safe to re-run manually if first boot fails partway:
#     podman exec postgres bash /docker-entrypoint-initdb.d/init.sh
set -euo pipefail

# ── Roles ──────────────────────────────────────────────────────────────────

# Postgres has no CREATE ROLE IF NOT EXISTS, so re-running this script would
# abort on the first existing role. CREATE-then-ALTER in a DO block is both
# idempotent and applies any password change from .env.
create_role() {
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-SQL
	DO \$\$
	BEGIN
	    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$1') THEN
	        CREATE ROLE $1 LOGIN;
	    END IF;
	END
	\$\$;
	ALTER ROLE $1 LOGIN PASSWORD '$2';
	SQL
}

create_role "${APP_DB_USER}"       "${APP_DB_PASSWORD}"
create_role "${SEAWEEDFS_DB_USER}" "${SEAWEEDFS_DB_PASSWORD}"
create_role "${PHOENIX_DB_USER}"   "${PHOENIX_DB_PASSWORD}"
create_role "${DAGSTER_DB_USER}"   "${DAGSTER_DB_PASSWORD}"
create_role "${ARGILLA_DB_USER}"   "${ARGILLA_DB_PASSWORD}"
create_role "${GRAFANA_DB_USER}"   "${GRAFANA_DB_PASSWORD}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -c "ALTER DATABASE ${POSTGRES_DB} OWNER TO ${APP_DB_USER}"

# ── Databases ──────────────────────────────────────────────────────────────

db_exists() {
    psql --username "$POSTGRES_USER" --dbname postgres -Atc \
        "SELECT 1 FROM pg_database WHERE datname = '$1'" | grep -q 1
}

db_exists "$SEAWEEDFS_DB_NAME" || psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE ${SEAWEEDFS_DB_NAME} OWNER ${SEAWEEDFS_DB_USER}"

db_exists "$PHOENIX_DB_NAME" || psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE ${PHOENIX_DB_NAME} OWNER ${PHOENIX_DB_USER}"

db_exists "$DAGSTER_DB_NAME" || psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE ${DAGSTER_DB_NAME} OWNER ${DAGSTER_DB_USER}"

db_exists "$ARGILLA_DB_NAME" || psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE ${ARGILLA_DB_NAME} OWNER ${ARGILLA_DB_USER}"

# ── Application database ───────────────────────────────────────────────────
# pgvector for embedding similarity; pg_trgm as a lexical prefilter.
# Application schema is managed by the app, not here.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    GRANT ALL ON SCHEMA public TO ${APP_DB_USER};
SQL

# ── Grafana's read-only view of the application database ───────────────────
# The "Pipeline state" dashboard counts queue depths and acceptance rates
# straight off the tables, because that is where the truth about them is.
# It reads as its own role: a dashboard is a place people paste SQL, and
# the application role can DROP.
#
# ALTER DEFAULT PRIVILEGES is the load-bearing line. This runs on an empty
# database - Alembic has not run yet - so GRANT SELECT ON ALL TABLES grants
# select on nothing. The default privilege is what covers every table the
# app role creates afterwards. Both are here: the ALTER for the tables to
# come, the GRANT for a database that already has them, which is what a
# re-run of this script is for.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    REVOKE ALL ON DATABASE ${POSTGRES_DB} FROM ${GRAFANA_DB_USER};
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${GRAFANA_DB_USER};
    GRANT USAGE ON SCHEMA public TO ${GRAFANA_DB_USER};
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${GRAFANA_DB_USER};
    ALTER DEFAULT PRIVILEGES FOR ROLE ${APP_DB_USER} IN SCHEMA public
        GRANT SELECT ON TABLES TO ${GRAFANA_DB_USER};
SQL

# ── SeaweedFS database ─────────────────────────────────────────────────────

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$SEAWEEDFS_DB_NAME" <<-SQL
    CREATE TABLE IF NOT EXISTS filemeta (
        dirhash   BIGINT,
        name      VARCHAR(65535),
        directory VARCHAR(65535),
        meta      BYTEA,
        PRIMARY KEY (dirhash, name)
    );
    ALTER TABLE filemeta OWNER TO ${SEAWEEDFS_DB_USER};
    GRANT ALL ON SCHEMA public TO ${SEAWEEDFS_DB_USER};
SQL

# ── Phoenix database ───────────────────────────────────────────────────────

# Grafana reads it too, for the run page that puts a run's spans beside its
# rows. Phoenix owns this schema, so the default privilege is for its role.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$PHOENIX_DB_NAME" <<-SQL
    GRANT ALL ON SCHEMA public TO ${PHOENIX_DB_USER};
    GRANT CONNECT ON DATABASE ${PHOENIX_DB_NAME} TO ${GRAFANA_DB_USER};
    GRANT USAGE ON SCHEMA public TO ${GRAFANA_DB_USER};
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${GRAFANA_DB_USER};
    ALTER DEFAULT PRIVILEGES FOR ROLE ${PHOENIX_DB_USER} IN SCHEMA public
        GRANT SELECT ON TABLES TO ${GRAFANA_DB_USER};
SQL

# ── Dagster database ───────────────────────────────────────────────────────
# Dagster manages its own schema via its migration tooling; we only grant access.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$DAGSTER_DB_NAME" <<-SQL
    GRANT ALL ON SCHEMA public TO ${DAGSTER_DB_USER};
SQL

# ── Argilla database ───────────────────────────────────────────────────────
# Argilla manages its own schema via Alembic on first start; we only grant access.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$ARGILLA_DB_NAME" <<-SQL
    GRANT ALL ON SCHEMA public TO ${ARGILLA_DB_USER};
SQL

echo "init: ${POSTGRES_DB}, ${SEAWEEDFS_DB_NAME}, ${PHOENIX_DB_NAME}, ${DAGSTER_DB_NAME}, ${ARGILLA_DB_NAME} ready"
echo "init: ${GRAFANA_DB_USER} may read ${POSTGRES_DB} and write nothing"
