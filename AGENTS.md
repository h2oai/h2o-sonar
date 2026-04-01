# Copilot instructions

This document provides instructions to the copilot AI assistants for h2oai/h2o-sonar project.

## General instructions

- Always write beautiful, readable, and maintainable code.
- Handle errors, exceptions, and corner cases.
- Prefer clarity over cleverness. Optimize only when needed and measured.
- Always KISS - keep changes small and focused.
- Always DRY the code - do not duplicate code; create reusable classes, functions and methods; do not repeat yourself.
- Always add tests alongside code changes.

## Functional architecture instructions

- Contribute to this repository which is the library for predictive and generative AI models evaluation of accuracy, explainability, fairness, and performance.

## Technology stack instructions

- The library is written in Python.
- Always use Python 3.11 and avoid newer language features.
- Always use Google code style imports - import module, not symbols (except typing).
- Always do global imports - never do local imports - unless you have really strong justification.
- Always use type hints in Python - use Python 3.11 style type hints like tuple and list, | instead of Optional.
- Always use keyword parameters if a function/method call has more than 1 argument - like `retval = myfun(par1=val1, par2=var2)`.
- Always use f-strings for string formatting.
- Always start code comments with lowercase letter.
- Always use `numpy` docstring convention.

## Code quality instructions

- Always use `make lint` to format and check code quality - it runs ruff for formatting, import sorting, and linting. Never run `ruff` individually.
- Code formatting, import sorting, and linting use `ruff` with 88 columns configured in `pyproject.toml`.
- ruff replaces black (formatting), isort (import sorting), and flake8 (linting) with a single, fast tool.

## Repository conventions

- Library code lives under `h2o_sonar/`.
- Tests code lives under `tests/` and mirrors the package structure.
- Datasets, configs, and evaluation assets live under `data/`.
- Examples and notebooks live under `examples/`.
- ReStructuredText documentation sources live under `docs/`.
- Python package licenses are stored in `licenses`.
- Python build tools licenses are stored in `licenses/tools-licenses`.
- Python datasets licenses are stored in `licenses/datasets-licenses`.

## Test instructions

- Always use `pytest` for testing - always use `uv` to run `pytest`.
- Each test (function) is structured into 3 sections: `# GIVEN`, `# WHEN`, and `# THEN`. `GIVEN` section prepares the data, `WHEN` section calls the function, and `THEN` section prints results, asserts results and checks results.
- `THEN` section of the test must have at least one assert statement.
- Name test files `test_*.py` and test functions `test_*`.
- Always use fixtures for test setup - avoid global state.
- Always mark generative tests with `@pytest.mark.generative`.
- Always use `@pytest.mark.slow` marker for long-running tests and gate them in CI.
- Keep tests deterministic.
- Always use text to indicate success/failure/progress like DONE, ERROR or WIP - never use (unicode) characters like ✓ or ✗.
- Always print or log intermediate values only when they aid debugging.
- Always make sure that tests which test new feature or fix are in green.

## Build instructions

- The project is built with `make` with the common targets defined in `Makefile`:
  - Always use `make help` to find out what are the targets to format, lint, build, package, and test the repository.
- Use `uv` for Python dependency management:
  - Dependencies are defined in `pyproject.toml` under `[project.dependencies]` and `[project.optional-dependencies]`.
  - The project has package extras - `evaluators`, `explainers`, `genaiclient` - each with its own dependencies defined in `pyproject.toml`.
  - When adding a dependency, pin to the latest version.
- Consider `uv` as the primary installer.
- Always use `uv` if you need to format, lint, build or test project without `make`.

## Documentation instructions

- ReStructuredText documentation sources are in `docs/` directory.
- Always use `Makefile` targets to build the ReStructuredText documentation.
- Use NumPy-style docstrings for public functions, classes, and modules in Python.
- Keep examples runnable - prefer doctests when practical.

## Continuous Integration instructions

- GitHub Actions is used as CI.
- GitHub Actions CI configuration is stored under `.github/workflows/`.
- Pin action versions to commit SHAs or version tags.

## Security and secrets instructions

- Always use environment variables and secret stores.
- Always use GitHub actions secrets.
- Always maintain test infrastructure details outside of files stored in Git - use tests/lib/given_generative.json and related Python module.
- Never commit secrets, credentials or sensitive data.
- Validate, sanitize and anonymize all external inputs.
- Always run security-focused checks using tools like `pip-audit`.
- Always add Python package license to `licenses/` when you add new direct dependency.

## Release versioning instructions

- Always use semantic versioning: MAJOR.MINOR.PATCH.
- Note that releases has Git tag like `vMAJOR.MINOR.PATCH`.
- Note that releases are being developed in `dev-MAJOR.MINOR.PATCH` branches.
- Note that Git branches use naming convention for fix branch (`bug-NUMBER/DESCRIPTION`), features and enhancements (`feat-NUMBER/DESCRIPTION`) and documentation (`doc-NUMBER/DESCRIPTION`).
- Note that Conventional commits (conventionalcommits.org) are used for the commit messages.
- Always update change log stored in `CHANGELOG.md` whenever you do a fix, change, or enhancement.
- Always make sure that the version is consistent in `README.md`, `version.py`, `index.rst` and `CHANGELOG.md` - `version.py` is the one and only authoritative version source.
