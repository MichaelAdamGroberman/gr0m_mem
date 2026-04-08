"""Allow `python -m gr0m_mem` to run the CLI."""

from gr0m_mem.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
