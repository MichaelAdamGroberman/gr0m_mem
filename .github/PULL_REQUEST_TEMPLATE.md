<!--
  Thanks for the pull request! Before you submit, please check off
  every item in the "Required checks" list. If anything is unchecked,
  the maintainer will ask for it — saving a round trip just means
  filling the list out now.
-->

## What does this change?

<!-- 1-3 sentences describing the user-visible effect. -->

## Why?

<!-- Link the issue this fixes, or describe the problem if there is no issue. -->

Closes #

## How did you test it?

<!-- Commands you ran, tests you added, before/after output. -->

```console
$ .venv/bin/python -m pytest -q tests/unit/test_<your_area>.py

```

## Required checks

- [ ] The three CI commands pass locally:
  - [ ] `ruff check gr0m_mem tests benchmarks`
  - [ ] `mypy gr0m_mem benchmarks`
  - [ ] `pytest -q`
- [ ] If this touches the KG, loop-prevention, or retrieval surfaces, the loop-prevention benchmark still passes in strict mode:
  - [ ] `python -m benchmarks.loop_prevention.run --strict`
- [ ] New public API is covered by a test in `tests/unit/`, `tests/temporal/`, or `tests/integration/`
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)
- [ ] If this changes behavior users will notice, `CHANGELOG.md` has an entry under `## [Unreleased]`
- [ ] I have read the [Code of Conduct](CODE_OF_CONDUCT.md) and I am willing to follow it

## Optional

- [ ] This is a breaking change
- [ ] This adds a new dependency (justify in the description)
- [ ] This changes the on-disk database schema (describe the migration)
