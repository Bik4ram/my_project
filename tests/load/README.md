# Tests

Load-test scripts for SemRAG. Not part of the unit-test suite —
invoked manually against a running API.

- `locustfile.py` — Locust scenarios. The realistic option for
  stress-testing. Run with:

  ```bash
  pip install locust
  locust -f tests/load/locustfile.py --host http://localhost:8000
  ```

- `run_load_test.py` — dependency-free smoke load test. No
  external deps beyond `requests`. Run with:

  ```bash
  python -m tests.load.run_load_test --url http://localhost:8000 \
      --concurrency 4 --duration 30 --json
  ```

The two scripts overlap intentionally — Locust is the
production-grade option; the standalone script is the
no-installs option for CI smoke tests.