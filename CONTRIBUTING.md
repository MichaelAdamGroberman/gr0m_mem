# Contributing to Gr0m_Mem

Thanks for your interest! This file is the practical "how do I help"
guide. The governance contract is in
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) and the disclosure policy
is in [`SECURITY.md`](SECURITY.md).

## Before you open an issue or PR

- **Security issues** — please do NOT open a public issue. Use the
  GitHub private vulnerability reporting button on the Security tab,
  or email the maintainer. See [`SECURITY.md`](SECURITY.md) for the
  full disclosure policy.
- **Bug reports** — use the **Bug report** issue template
  (`.github/ISSUE_TEMPLATE/bug_report.yml`). Include:
  - Gr0m_Mem version (`gr0m_mem --version`)
  - Python version (`python --version`)
  - OS and arch (`uname -a` on Unix)
  - Backend selected (`gr0m_mem doctor`)
  - A minimal reproducer
  - The actual error or unexpected behavior
- **Feature requests** — use the **Feature request** template. Tell us
  the problem you're trying to solve before the solution you have in
  mind. We're more likely to merge a small change that addresses a
  real pain point than a big change that adds options.
- **Questions** — open a GitHub Discussion instead of an issue.

## Local development setup

```bash
# Clone your fork
git clone https://github.com/<you>/gr0m_mem.git
cd gr0m_mem

# Use uv (fast) — recommended
uv venv --python 3.12
uv pip install -e ".[dev]"

# Or use plain pip + venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks locally

The CI runs the exact same three commands, so if these pass locally
the PR will pass CI:

```bash
.venv/bin/python -m ruff check gr0m_mem tests benchmarks
.venv/bin/python -m mypy gr0m_mem benchmarks
.venv/bin/python -m pytest -q
```

For the loop-prevention benchmark (required to stay at 100%):

```bash
.venv/bin/python -m benchmarks.loop_prevention.run --strict
```

## Coding conventions

- **Python 3.10+**, `from __future__ import annotations` at the top of
  every file
- **Ruff** with the lint rules in `pyproject.toml` (E, F, I, B, UP, SIM).
  No ignores without a comment explaining why
- **Mypy `--strict`** — every function signature gets type hints, no
  `Any` leakage across module boundaries
- **Conventional Commits** for commit messages (`feat:`, `fix:`,
  `docs:`, `test:`, `ci:`, `refactor:`, `perf:`, `chore:`, `sec:`,
  optionally scoped like `feat(store):`)
- **Tests first** for any non-trivial change. Public API changes need
  tests in `tests/unit/` (or `tests/temporal/` for KG changes, or
  `tests/integration/` for CLI/hook round trips)
- **No LLM calls from inside the library** — Gr0m_Mem produces context
  blocks, it does not reason about them. See the rationale in
  [`README.md`](README.md)

## Pull request flow

1. Open an issue first if the change is non-trivial — "non-trivial" =
   anything that adds or removes a public symbol, changes the on-disk
   format, or touches the MCP tool surface
2. Fork, branch (`feat/short-topic` or `fix/short-topic`), make the
   change, make sure the three local checks pass
3. Open the PR using the template (`.github/PULL_REQUEST_TEMPLATE.md`)
4. CI runs across macOS + Ubuntu on Python 3.10, 3.11, 3.12. All three
   matrix cells plus the loop-prevention benchmark must be green
5. The maintainer is `@MichaelAdamGroberman` and is auto-requested as
   a reviewer via `CODEOWNERS`

## License

By contributing you agree that your contribution will be licensed
under the [MIT License](LICENSE) that covers this project.
