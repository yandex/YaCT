"""Allow ``python -m lib`` from a checkout (repo root on PYTHONPATH)."""

from lib.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
