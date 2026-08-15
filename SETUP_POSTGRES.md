# Local Postgres setup for the test suite

```bash
# Install & start Postgres 16 (Ubuntu)
sudo apt-get install -y postgresql-16 postgresql-16-pgvector
sudo service postgresql start

# Create role + test database (password must match the test-suite default)
sudo -u postgres psql -c "CREATE USER semrag WITH PASSWORD 'semrag_test' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE semrag_test OWNER semrag;"

# Apply migrations (creates pgvector extension + chunks table too)
export SEMRAG_DATABASE_URL="postgresql+psycopg2://semrag:semrag_test@localhost:5432/semrag_test"
alembic upgrade head

# Run the storage + vector tests
export SEMRAG_TEST_DATABASE_URL="$SEMRAG_DATABASE_URL"
PYTHONPATH=src pytest tests/unit/storage tests/unit/vector -v
```

If no Postgres is reachable, these test fixtures skip gracefully
(see `tests/unit/conftest.py`) — the rest of the suite (Parts 2-4) is
unaffected.

## Qdrant (benchmark only, not required to run the test suite)

An earlier benchmark compared pgvector against a locally-run prebuilt
Qdrant binary. Not a dependency of the shipped code (pgvector was the
V1 choice); that comparison is historical and not repeated — see
[`docs/CODEBASE_GUIDE.md`](./docs/CODEBASE_GUIDE.md) and
[`src/semrag/storage/README.md`](./src/semrag/storage/README.md) for the
storage rationale.
