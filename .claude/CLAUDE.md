# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

H2O Sonar is a comprehensive AI explainability and evaluation library with two major components:

1. **Predictive AI (Explainers)**: Provides 18 explainers for traditional ML models (H2O-3, Driverless AI MOJO, scikit-learn)
2. **Generative AI (Evaluators)**: Provides 44 evaluators for LLM/RAG systems (h2oGPTe, OpenAI, Azure, Bedrock, ollama)

### Core Module Structure

```
h2o_sonar/
├── explainers/          # Predictive AI explainer implementations (18 explainers)
├── evaluators/          # Generative AI evaluator implementations (44 evaluators)
├── lib/                 # Core library infrastructure
│   ├── api/            # Public API layer (interpretations, evaluations, results)
│   ├── container/      # Explainer/evaluator container and registry
│   └── integrations/   # External integrations (Generative AI clients)
├── methods/            # Low-level algorithms (SHAP, PD/ICE, surrogates, NLP)
├── utils/              # Utilities (caching, sampling, crypto, I/O)
├── models/             # Model wrappers and abstractions
├── interpret.py        # Entry point for predictive AI interpretations
├── evaluate.py         # Entry point for generative AI evaluations
└── config.py           # Configuration management (GPU/CPU, devices, hosts)
```

### Key Design Patterns

**Explainer/Evaluator Container Pattern**: All explainers and evaluators are registered in a container (`lib/container/explainer_container.py`) and accessed via descriptors. Each implements a common interface defined by abstract base classes.

**BYOE (Bring Your Own Explainer/Evaluator)**: Custom explainers/evaluators can be added via Python recipes. Templates and examples are in `tests/explainers/examples/` and `tests/explainers/templates/`.

**Dual Entry Points**:
- `interpret.py` for predictive AI model explanations
- `evaluate.py` for generative AI model evaluations

**Model Abstraction Layer**: Models are wrapped in adapters (`models/`) to provide uniform interface across different backends (MOJO, H2O-3, scikit-learn, LLMs).

**Configuration System**: GPU/CPU device selection and host configurations managed through `config.py` with environment variable overrides (`H2O_SONAR_CFG_DEVICE`).

## Python Code Style

**General Principles**:
- Write beautiful, readable, and maintainable code
- Handle errors, exceptions, and corner cases
- Prefer clarity over cleverness - optimize only when needed and measured
- KISS - keep changes small and focused
- DRY - do not duplicate code; create reusable classes, functions and methods

**Python 3.11 Requirements**:
- Use Python 3.11 syntax and features
- Avoid Python 3.12+ features for compatibility

**Import Style**:
- Use Google code style imports - import module, not symbols (except typing)
- Example: `import numpy as np` instead of `from numpy import array`
- Imports sorted by isort with `.isort.cfg` configuration

**Type Hints**:
- Always use type hints in Python
- Use Python 3.11 style: `tuple`, `list`, `|` instead of `Optional`
- Example: `def func(x: int | None) -> list[str]:`

**String Formatting**:
- Always use f-strings for string formatting
- Example: `f"Value: {x}"`

**Comments**:
- One line comments: always start with lowercase letter except when there is a word w/ all uppercase letters like GIVEN
- Use `#` for inline comments

**Docstrings**:
- Always use NumPy docstring convention for public functions, classes, and modules
- Keep examples runnable - prefer doctests when practical

## Common Commands

### Environment Setup
```bash
# Check development environment (Python, uv, git, Java, GPU, etc.)
make diagnostics

# Print make configuration
make print_make_config

# Create/recreate virtual environment (Python 3.11 default)
make setup

# Change Python version (e.g., Python 3.10)
make setup TARGET_PYTHON_VERSION=3.10
```

### Build and Install
```bash
# Build the project (non-cythonized)
make build

# Create source wheel
make dist_src

# Create binary (cythonized) wheel
make dist

# Install from wheel
make install

# Uninstall H2O Sonar
make uninstall
```

### Dependency Management
```bash
# Download private dependencies from S3 (requires AWS credentials)
make private_deps

# Install all dependencies (explainers + evaluators)
make install_deps

# Install only generative AI client dependencies
make install_deps_generative_client

# Install test dependencies
make install_test_deps

# Install analysis dependencies (linters, formatters)
make install_analysis_deps
```

### Testing
```bash
# Run all tests (predictive + generative AI)
make test

# Run predictive AI smoke tests (fast)
make test_predictive_smoke

# Run predictive AI tests (skips long-running SHAP tests)
make test_predictive

# Run generative AI tests (CPU mode)
make test_generative

# Run generative AI smoke tests (quick sanity check)
make test_generative_smoke

# Run generative AI GPU-accelerated evaluator tests
make test_generative_gpu

# Run generative AI GPU tests with different configs (auto/cpu/gpu)
make test_generative_gpu_cfg

# Run generative AI client extras tests
make test_generative_client

# Test cythonization on distribution wheel
make test_dist

# Run test coverage
make test-coverage

# Run a single test
pytest tests/lib/test_evaluate.py::test_all_evaluators

# Run tests with verbose output and fail fast
pytest -svv --maxfail=1 tests/lib/test_evaluator_bias_toxicity_hallucinations.py

# Run tests matching pattern
pytest -k "test_sanity" tests/lib/
```

### Code Quality

**IMPORTANT**: Always use `make lint` to format and check code quality. Never run `ruff` individually.

```bash
# Run all linters and formatters (precommit hook: lint + security)
make precommit

# Format and lint both code and test code
make lint

# Source code operations
make code_format               # Format and fix with ruff (88 columns)
make code_format_check         # Check ruff formatting (required)
make code_analysis             # Lint with ruff (required)
make code_analysis_mypy        # Type check with mypy
make code_compile              # Compile to check syntax errors (required)

# Test code operations
make test_code_format          # Format and fix test code with ruff
make test_code_format_check    # Check test formatting (required)
make test_code_analysis        # Lint test code with ruff (required)
make test_code_analysis_mypy   # Type check test code
make test_compile              # Compile test code (required)

# Security
make code_security             # Security audit with pip-audit
```

**Configuration**:
- ruff: 88 line length, Python 3.11 target, combines formatting (black-compatible), import sorting (isort-compatible), and linting (flake8-compatible) (configured in `pyproject.toml`)
- mypy: Python 3.11, ignores missing imports (configured in `pyproject.toml`)

### Documentation
```bash
# Build ReStructuredText documentation with Sphinx
make documentation

# Build documentation without reinstalling dependencies (faster)
make documentation-fast

# Build documentation for H2O Eval Studio variant
make documentation_es

# Generate changelog from CHANGELOG.md
# (automatically run by documentation target)
pandoc --from=markdown --to=rst --output=docs/source/changelog.rst ./CHANGELOG.md

# Generate H2O Eval Studio test classes configuration
make h2o_es_test_class_cfg

# Generate licenses CSV overview
make licenses_csv

# View documentation - look for output like:
# file:///path/to/h2o-sonar/docs/build/html/index.html
```

### Cleaning
```bash
# Remove build artifacts
make clean

# Remove dependencies
make clean_deps

# Clean all (build + deps)
make purge

# Clean everything including git-untracked files
make mrproper

# Rename cached models directories (HuggingFace, H2O Sonar, NLTK)
make clean_models_cache
```

### Cython Operations

For testing cythonized builds:

```bash
# Build, install cythonized wheel, and hide source directory
make cython_to

# Restore source directory and uninstall
make cython_from

# Install source wheel and hide sources
make src_to
```

### Package Manager (uv)

**uv** is the primary package installer (modern Python package manager):

```bash
# Check if uv is installed
make check_uv

# Manual installation with uv
uv pip install -e ".[explainers,evaluators]"

# Install specific dependency groups
uv pip install --group test      # Test dependencies
uv pip install --group style     # Linters/formatters
uv pip install --group docs      # Documentation tools
uv pip install --group notebooks # Jupyter Lab

# Run Python with uv
uv run --with ipython ipython
```

**Requirements**:
- uv >= 0.4.0 recommended
- Supports dependency-groups (PEP 735)

## Testing Patterns

**Structure**: Tests follow a consistent 3-section structure:

```python
def test_example():
    # GIVEN - setup test data and preconditions
    dataset = load_test_dataset()
    model = create_test_model()

    # WHEN - execute the function under test
    result = run_interpretation(dataset=dataset, model=model)

    # THEN - assert expected outcomes, print results, check results
    print(f"Result: {result}")  # print intermediate values only when they aid debugging
    assert result is not None
    assert result.status == "success"
```

**Requirements**:
- Test files: `test_*.py`
- Test functions: `test_*`
- THEN section MUST have at least one assert statement
- Always use fixtures for test setup - avoid global state
- Keep tests deterministic
- Always add tests alongside code changes
- Always make sure that tests which test new feature or fix are in green

**Pytest Markers**:
- `@pytest.mark.generative` - Generative AI tests for LLMs and RAGs
- `@pytest.mark.slow` - Long-running tests (gate them in CI)
- `@pytest.mark.flaky` - Flaky test - may fail intermittently
- `@pytest.mark.expensive` - Expensive test - preferably not to be run on CI
- `@pytest.mark.agentic` - Testing agents or agentic evaluations
- `@pytest.mark.h2ogpte` - h2oGPTe test
- Other markers defined in `pyproject.toml` [tool.pytest.ini_options]

**Best Practices**:
- Use text to indicate success/failure/progress like DONE, ERROR or WIP
- Never use unicode characters like ✓ or ✗ in test output
- Print or log intermediate values only when they aid debugging
- Always use `pytest` with `uv` for testing

**H2O-3 Cleanup Pattern**:

Two consistent patterns for H2O-3 cleanup - choose based on test style:

1. **Pytest-style tests (functions)** → Use `h2o3_cleanup_fixture` parameter
2. **Unittest-style tests (classes)** → Inherit from `BaseH2OTest`

Never use try-finally for H2O-3 resource management.

**Pytest-style example:**
```python
# CORRECT - Pytest-style with fixture
def test_h2o_example(tmpdir, h2o3_cleanup_fixture):
    # test code here
    # cleanup happens automatically after test

# INCORRECT - Don't use try-finally
def test_h2o_example_bad(tmpdir):
    try:
        # test code
    finally:
        h2o_utils.clean_up_h2o3()  # DON'T DO THIS
```

**Unittest-style example:**
```python
# CORRECT - Unittest-style with BaseH2OTest
from tests.base_h2o_test import BaseH2OTest

class TestMyH2OFeature(BaseH2OTest):
    def setUp(self):
        super().setUp()
        # setup code

    def test_something(self):
        # test code here
        # cleanup happens automatically after each test

# INCORRECT - Don't use manual tearDown
class TestMyH2OFeatureBad(unittest.TestCase):
    def tearDown(self):
        h2o_utils.clean_up_h2o3()  # DON'T DO THIS
```

**Additional guidelines:**
- Use pytest's built-in `tmpdir` fixture instead of `tempfile.mkdtemp()`
- Session fixture `h2o3_init_fixture` handles cluster lifecycle (startup)
- Session fixture `h2o3_shutdown_at_end` handles cluster shutdown (end of tests)
- Cleanup fixture `h2o3_cleanup_fixture` removes frames/models after each test
- BaseH2OTest provides both `tearDown()` and `tearDownClass()` for automatic cleanup

**Target Coverage**: 80%+

## GPU/CPU Configuration

Evaluators can run on CPU or GPU. Control via environment variable:

```bash
# Force CPU mode (default for CI)
H2O_SONAR_CFG_DEVICE="cpu" pytest tests/lib/test_evaluator_perplexity.py

# Force GPU mode
H2O_SONAR_CFG_DEVICE="gpu" pytest tests/lib/test_evaluator_perplexity.py

# Auto-detect (default)
H2O_SONAR_CFG_DEVICE="" pytest tests/lib/test_evaluator_perplexity.py
```

GPU-accelerated evaluators include: BERTScore, Answer Semantic Similarity, GPTScore, Perplexity, Toxicity, Hallucination.

## Package Extras

The project has modular extras for different use cases:

```bash
# Install all features
pip install h2o_sonar-<version>.whl[explainers,evaluators]

# Predictive AI explainers only
pip install h2o_sonar-<version>.whl[explainers]

# Generative AI evaluators only
pip install h2o_sonar-<version>.whl[evaluators]

# Generative AI client only (minimal footprint)
pip install h2o_sonar-<version>.whl[genaiclient]

# Development dependencies
uv pip install --group dev        # All dev dependencies
uv pip install --group test       # Test dependencies
uv pip install --group style      # Linters and formatters
uv pip install --group docs       # Documentation dependencies
uv pip install --group notebooks  # Jupyter Lab dependencies
```

**Configuration**:
- Project dependencies: `pyproject.toml` under `[project.dependencies]`
- Optional dependencies: `pyproject.toml` under `[project.optional-dependencies]`
- Development dependency groups: `pyproject.toml` under `[dependency-groups]` (PEP 735)
- When adding a dependency, pin to the latest version
- Use `uv` as the primary package installer

## Examples and Jupyter Notebooks

```bash
# Install H2O Sonar and start Jupyter Lab
make examples_jupyter_lab

# Manually install Jupyter dependencies
make install_jupyter_deps

# Install H2O Sonar in editable mode for notebooks
make install_h2o_sonar_for_jupyter
```

**Location**:
- Examples: `examples/`
- Notebooks: `examples/*.ipynb`
- Predictive BYOE examples: `tests/explainers/examples/`
- Predictive BYOE templates: `tests/explainers/templates/`

## Security and Secrets

**Best Practices**:
- Never commit secrets, credentials or sensitive data
- Always use environment variables and secret stores
- Always use GitHub Actions secrets for CI
- Validate, sanitize and anonymize all external inputs
- Always run security-focused checks using `pip-audit` via `make code_security`
- Always add Python package license to `licenses/` when adding new direct dependency

**Test Infrastructure**:
- Maintain test infrastructure details outside of Git
- Use `tests/lib/given_generative.json` for local configuration (not in Git)
- Template downloaded from S3: `tests/lib/given_generative_TEMPLATE.json`
- Download with: `make private_test_config`

**Private Data**:
```bash
# Download private configuration from S3
make private_test_config

# Download private test data from S3
make private_test_data
```

## Release Versioning

**Semantic Versioning**: MAJOR.MINOR.PATCH

**Version Files**:
- `version.py` - One and only authoritative version source
- Must be consistent across: `README.md`, `version.py`, `index.rst`, `CHANGELOG.md`

**Git Conventions**:
- Git tags: `vMAJOR.MINOR.PATCH`
- Development branches: `dev-MAJOR.MINOR.PATCH`
- Fix branches: `bug-NUMBER/DESCRIPTION`
- Feature/enhancement branches: `feat-NUMBER/DESCRIPTION` or `enh-NUMBER/DESCRIPTION`
- Documentation branches: `doc-NUMBER/DESCRIPTION`

**Commit Messages**:
- Use Conventional Commits (conventionalcommits.org)
- Examples: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`

**Changelog**:
- Always update `CHANGELOG.md` whenever you do a fix, change, or enhancement
- Generate new changelog section: `make releng_changelog_md_new`

**Release Commands**:
```bash
# Update version in sources and documentation
make src_update_version LAST_VERSION=x.y.z NEXT_VERSION=x.y.z

# Create source wheel
make release_whl_src

# Create binary wheel
make release_whl_bin

# Create documentation zip
make release_doc_zip
```

## Continuous Integration

**Platform**: GitHub Actions

**Configuration**: `.github/workflows/`

**Best Practices**:
- Pin action versions to commit SHAs or version tags
- Use GitHub Actions secrets for sensitive data

## Platform Notes

- **Linux x86_64**: Full support (MOJO, REST, H2O-3, scikit-learn)
- **macOS x86_64**: Limited support (REST, H2O-3, scikit-learn; no MOJO)
- **Python 3.11**: Primary supported version (avoid Python 3.11+ features)
- **Python 3.10**: Secondary support via virtualenv

## Development Workflow

**Branching Strategy**:
- Work on feature branches, never commit directly to main
- Branch naming: `feat-NUMBER/description`, `bug-NUMBER/description`, `doc-NUMBER/description`

**Before Committing**:
- Run `make precommit` (linting + security audit)
- Ensure all tests pass
- Add tests alongside code changes
- Update `CHANGELOG.md` for fixes, changes, or enhancements

**Pull Requests**:
- Rebase or squash merge PRs (no merge commits)
- Follow Conventional Commits for messages

**Build Types**:
- Development: Source wheels (non-cythonized) via `make dist_src`
- Production: Binary wheels (cythonized) via `make dist`

**Tools**:
- Use `uv` for dependency management
- Use `make` for all build, test, and quality operations
- CI runs on GitHub Actions (`.github/workflows/`)

## Running Single Tests

**IMPORTANT**: Always activate virtual environment first: `. .venv/bin/activate`

```bash
# Run specific test by path and name
pytest -sv tests/lib/test_evaluate.py::test_all_evaluators

# Run with verbose output and fail fast
pytest -svv --maxfail=1 tests/lib/test_evaluator_bias_toxicity_hallucinations.py

# Run tests matching pattern
pytest -k "test_sanity" tests/lib/

# Run with specific markers
pytest -m generative tests/           # Only generative AI tests
pytest -m "not generative" tests/     # Only predictive AI tests
pytest -m slow tests/            # Only slow tests

# Run with coverage
pytest --cov=h2o_sonar --cov-report=html tests/

# Run with custom options
pytest -ra --maxfail=10 tests/   # Show summary, stop after 10 failures
```

## Private Data and Configuration

The project uses private S3 buckets for:
1. Private dependencies (H2O Model Validation, h2oGPTe client)
2. Test configuration (infrastructure details)
3. Large test datasets

**Setup**:
```bash
# Download private dependencies (requires AWS credentials)
make private_deps

# Download private test configuration
make private_test_config

# Download large test datasets
make private_test_data
```

**Files Created**:
- `deps/$(MV_WHEEL)` - H2O Model Validation wheel
- `deps/$(H2OGPTE_CLIENT_WHEEL)` - h2oGPTe client wheel
- `tests/lib/given_generative_TEMPLATE.json` - Test config template
- `tests/lib/given_generative.json` - Your local test config (not in Git)
- `data/generative/eval_s3/` - Large test datasets

## Repository Structure

```
h2o-sonar-FLOSS/
├── h2o_sonar/           # Library code
├── tests/               # Test code (mirrors package structure)
├── data/                # Datasets, configs, and evaluation assets
├── examples/            # Examples and Jupyter notebooks
├── docs/                # ReStructuredText documentation sources
├── licenses/            # Python package licenses
│   ├── tools-licenses/      # Python build tools licenses
│   └── datasets-licenses/   # Python datasets licenses
├── .github/workflows/   # GitHub Actions CI configuration
├── make/                # Build scripts and utilities
├── pyproject.toml       # Project metadata and dependencies
├── setup.py             # Build configuration
├── .flake8              # Flake8 configuration (source code)
├── .flake8-test         # Flake8 configuration (test code)
├── .isort.cfg           # isort configuration
├── Makefile             # Build automation
├── CHANGELOG.md         # Release changelog
├── CLAUDE.md            # This file - Claude Code instructions
└── AGENTS.md            # Copilot instructions
```

## Quick Reference

**Get Help**:
```bash
make help                # Print all available make targets
make diagnostics         # Check development environment setup
```

**Development Cycle**:
```bash
# 1. Setup
make setup               # Create virtual environment
. .venv/bin/activate     # Activate it

# 2. Install dependencies
make install_deps        # Install all dependencies

# 3. Make changes to code
# ... edit files ...

# 4. Format and lint
make lint                # Format imports, code, tests + lint all

# 5. Test
make test_predictive     # Test predictive AI
make test_generative     # Test generative AI

# 6. Before commit
make precommit           # Lint + security audit

# 7. Commit
git add .
git commit -m "feat: your change description"
```

**Common Issues**:
- Virtual environment not active: Run `. .venv/bin/activate`
- Missing dependencies: Run `make install_deps`
- Linting failures: Run `make lint` to auto-fix
- Test failures: Check logs, ensure dependencies installed

## Rules

Always run the full relevant test suite after making changes. Never declare work 'done' or claim tests pass without actually executing the tests and verifying the output. If tests fail, fix ALL failures before reporting completion.

When fixing bugs or test failures, fix ALL instances across the entire codebase in one pass. Use grep/glob to find all occurrences before making changes. Do not fix files one-at-a-time waiting for user to report the next failure.

Fix root causes, not symptoms. Do not add skip decorators, backward-compatibility shims, or suppression code unless explicitly asked. Prefer regenerating test data, updating APIs, and fixing the actual source of errors.

Do not create unnecessary temporary files, demo scripts, or marker files. Keep changes minimal and focused on what the user asked for. If something seems like over-engineering, it probably is.

## Project Overview

This is a Python project using pytest for testing, pyproject.toml for dependencies, and GitHub Actions for CI. The primary language is Python. When editing Makefiles, keep changes minimal and targete.

## CI / GitHub Actions

When making changes that affect CI (GitHub Actions workflows, dependencies in pyproject.toml, test configurations), check for ALL related files that may need updating. Search for duplicate dependency entries, version conflicts, and path references across all workflow files.

## Additional Project Information

See @AGENTS.md for detailed copilot instructions and development guidelines.
