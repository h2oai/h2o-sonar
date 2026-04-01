# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

VERSION=make/version.mk
include $(VERSION)

CONFIG=make/config.mk
include $(CONFIG)

ACTIVE_VENV := $(shell echo $$VIRTUAL_ENV)

UNAME := $(shell uname)


# uv ecosystem
UV := $(shell which uv 2>/dev/null)
# make parameter: directory with GitHub Actions build artifacts
GH_ACTIONS_ARTIFACTS_DIR := ""
DATASETS_DIR := ""
CURRENT_DIR := $(shell pwd)
MODELS_INDEX := ""
# make parameter: product name
PRODUCT_NAME := H2O Sonar
# make parameter: version update parameters
LAST_VERSION := ""
NEXT_VERSION := ""
# make parameter: H2O Sonar GitHub repository clone with gh-pages branch
H2O_SONAR_GH_PAGES_PATH := "/tmp"
# make parameter: h2oGPTe wheel repository: "s3" or "pypi"
H2O_SONAR_H2OGPTE_WHEEL_REPO := "pypi"

# constants
DOC_DELIMITER=".........................."


default: help


.PHONY: help
help:
	@echo " _   _ ____   ___    ____"
	@echo "| | | |___ \ / _ \  / ___|  ___  _ __   __ _ _ __"
	@echo "| |_| | __) | | | | \___ \ / _ \| '_ \ / _\` | '__|"
	@echo "|  _  |/ __/| |_| |  ___) | (_) | | | | (_| | |"
	@echo "|_| |_|_____|\___/  |____/ \___/|_| |_|\__,_|_|   $(BASE_VERSION)"
	@echo ""
	@echo "Targets:"
	@echo $(DOC_DELIMITER)
	@echo "help                      print this help"
	@echo "diagnostics               check development environment setup"
	@echo "print_make_config         check make configuration"
	@echo $(DOC_DELIMITER)
	@echo "setup                     (re)create virtual environment in .venv/ directory"
	@echo "                          to change python version set TARGET_PYTHON_VERSION (defaults to 3.11)"
	@echo "private_deps              download private dependencies from S3"
	@echo "private_test_config       download private configuration from S3"
	@echo "private_test_data         download private test data from S3"
	@echo "build                     build and cythonize the project"
	@echo $(DOC_DELIMITER)
	@echo "clean                     remove build relicts"
	@echo "clean_private             clean all private data"
	@echo "clean_deps                clean dependencies"
	@echo "clean_models_cache        remove cached models by renaming Huggingface/H2O Sonar/NLTK cache directories"
	@echo "mrproper                  purge ALL files untracked by Git, purge ccache"
	@echo $(DOC_DELIMITER)
	@echo "documentation             generate .rst documentation"
	@echo "                          set PRODUCT_NAME='Eval Studio' shell variable to change product name"
	@echo "documentation_fast        generate .rst documentation without installing dependencies"
	@echo "                          set PRODUCT_NAME='Eval Studio' shell variable to change product name"
	@echo "src_update_version        update H2O Sonar version in sources and documentation"
	@echo "                          set LAST_VERSION and NEXT_VERSION make variables to update version"
	@echo "h2o_es_test_class_cfg     generate test classes configuration for H2O Eval Studio"
	@echo "licenses_csv              generate overview of Python deps, versions, licenses and repos as CSV"
	@echo $(DOC_DELIMITER)
	@echo "dist                      create cythonized Python wheel to dist/"
	@echo "dist_src                  create source Python wheel to dist/"
	@echo "tarball                   create distribution tarball (.whl, doc, .ipynb) to dist/"
	@echo "binary_distribution       create binary distribution to dist/"
	@echo "documentation_distro      create .rst documentation distribution to dist/"
	@echo "gh_actions_distros        create distros to dist/ from GitHub Actions artifacts"
	@echo "                          set GH_ACTIONS_ARTIFACTS_DIR=path/to/artifacts shell variable"
	@echo $(DOC_DELIMITER)
	@echo "release_whl_bin           create binary wheel to dist/"
	@echo "release_whl_src           create source wheel to dist/"
	@echo "release_whl_src_to_s3     upload source wheel to S3"
	@echo "release_doc_zip           create documentation .zip to dist/"
	@echo "release_doc_zip_es        create documentation .zip for H2O Eval Studio to dist/"
	@echo "release_doc_to_ghpages    release documentation to GitHub Pages branch"
	@echo "                          set H2O_SONAR_GH_PAGES_PATH=path/to/gh-pages-repo make variable"
	@echo "release_generative        create release artifacts for H2O Eval Studio"
	@echo $(DOC_DELIMITER)
	@echo "install                   install H2O Sonar to the active environment"
	@echo "uninstall                 uninstall H2O Sonar and its requirements"
	@echo $(DOC_DELIMITER)
	@echo "code_format               format and fix source code using ruff"
	@echo "code_format_check         check that source code has ruff format (required)"
	@echo "code_analysis             lint H2O Sonar sources using ruff check (required)"
	@echo "code_analysis_mypy        lint H2O Sonar sources using mypy"
	@echo "code_compile              compile source code to check for syntax errors (required)"
	@echo "code_security             check H2O Sonar using pip-audit"
	@echo $(DOC_DELIMITER)
	@echo "test                      run ALL tests"
	@echo "test_predictive_smoke     run smoke predictive AI smoke tests"
	@echo "test_predictive           run all tests except (long running) SHAP tests"
	@echo "test_generative           run generative AI (LLM / RAG) tests"
	@echo "test_generative_smoke     run generative AI (LLM / RAG) smoke tests"
	@echo "test_generative_gpu       run generative AI tests of GPU accelerated evaluators"
	@echo "test_generative_client    run generative AI client (extras) smoke tests"
	@echo "test_code_format          format and fix test code using ruff"
	@echo "test_code_format_check    check that test code has ruff format (required)"
	@echo "test_code_analysis        lint test code using ruff check (required)"
	@echo "test_code_analysis_mypy   lint test code using mypy"
	@echo "test_compile              compile test code to check for syntax errors (required)"
	@echo $(DOC_DELIMITER)
	@echo "test-coverage             run test coverage with reports in tests/build"
	@echo "random_attack             run random attack test with optional parameters:"
	@echo "                          DATASETS_DIR=path/datasets MODELS_INDEX=path/to/models.json"
	@echo $(DOC_DELIMITER)
	@echo "cython_to                 clean, build wheel, install it and hide h2o_sonar source dir"
	@echo "cython_from               move h2o_sonar dir from attic, uninstall h2o-sonar"
	@echo $(DOC_DELIMITER)
	@echo "lint                      sort imports, format and lint both code and test code"
	@echo "precommit                 lint and security audit"
	@echo $(DOC_DELIMITER)
	@echo "examples_jupyter_lab      install H2O Sonar and start Jupyter Lab to run examples"
	@echo $(DOC_DELIMITER)
	@echo "suite_lib_download        download evaluation library test suites from S3"
	@echo "suite_lib_clean           delete evaluation library test suites"
	@echo "suite_lib_gen_markdown    generate Markdown index for the evaluation test suite library"
	@echo "suite_lib_gen_html        generate HTML index for the evaluation test suite library"


########
# CONFIG
########


.PHONY: print_make_config
print_make_config:
	@echo "BASE_VERSION=$(BASE_VERSION)"
	@echo "NPROCS=$(NPROCS)"
	@echo "TARGET_PYTHON_VERSION=$(TARGET_PYTHON_VERSION)"
	@echo "PYTHON=$(PYTHON)"
	@echo "PYTHON_VERSION_FULL=$(PYTHON_VERSION_FULL)"
	@echo "PYTHON_CP_VERSION=$(PYTHON_CP_VERSION)"
	@echo "CYTHON_VERSION=$(CYTHON_VERSION)"
	@echo "CXX=$(CXX)"
	@echo "C_CACHE=$(C_CACHE)"
	@echo "DATATABLE_WHEEL_URL=$(DATATABLE_WHEEL_URL)"
	@echo "MV_WHEEL=$(MV_WHEEL)"
	@echo "H2O_SONAR_WHEEL_FILE=$(H2O_SONAR_WHEEL_FILE)"
	@echo "H2O_SONAR_WHEEL_PATH=$(H2O_SONAR_WHEEL_PATH)"
	@echo "H2O_SONAR_SRC_WHEEL_FILE=$(H2O_SONAR_SRC_WHEEL_FILE)"
	@echo "H2O_SONAR_SRC_WHEEL_PATH=$(H2O_SONAR_SRC_WHEEL_PATH)"
	@echo "UV=$(UV)"


#############
# DIAGNOSTICS
#############


.PHONY: diagnostics
diagnostics:
	@echo " _   _ ____   ___    ____"
	@echo "| | | |___ \ / _ \  / ___|  ___  _ __   __ _ _ __"
	@echo "| |_| | __) | | | | \___ \ / _ \| '_ \ / _\` | '__|"
	@echo "|  _  |/ __/| |_| |  ___) | (_) | | | | (_| | |"
	@echo "|_| |_|_____|\___/  |____/ \___/|_| |_|\__,_|_|   (v$(BASE_VERSION))"
	@echo ""
	@echo "System:"
	@echo $(DOC_DELIMITER)
	@printf "OS/Arch:              "
	@if [ "$(UNAME)" = "Linux" ] || [ "$(UNAME)" = "Darwin" ]; then \
		echo "\033[32m✓\033[0m $(UNAME) $(ARCH)"; \
	else \
		echo "\033[33m⚠ \033[0m $(UNAME) $(ARCH)"; \
	fi
	@echo "CPU cores:            $(NPROCS)"
	@printf "Disk space:           "
	@DISK_AVAIL=$$(df -h . 2>/dev/null | tail -1 | awk '{print $$4}'); \
	echo "$$DISK_AVAIL available"
	@echo ""
	@echo "Required:"
	@echo $(DOC_DELIMITER)
	@printf "Python:               "
	@if command -v python3 >/dev/null 2>&1; then \
		PYTHON_VER=$$(python3 --version 2>&1 | cut -d' ' -f2); \
		PYTHON_MAJOR=$$(echo $$PYTHON_VER | cut -d'.' -f1); \
		PYTHON_MINOR=$$(echo $$PYTHON_VER | cut -d'.' -f2); \
		if [ "$$PYTHON_MAJOR" = "3" ] && [ "$$PYTHON_MINOR" = "11" ]; then \
			echo "\033[32m✓\033[0m $$PYTHON_VER"; \
		elif [ "$$PYTHON_MAJOR" = "3" ] && [ "$$PYTHON_MINOR" -ge "10" ]; then \
			echo "\033[33m⚠ \033[0m $$PYTHON_VER (recommend 3.11)"; \
		else \
			echo "\033[31m✗\033[0m $$PYTHON_VER (need 3.11)"; \
		fi; \
	else \
		echo "\033[31m✗\033[0m NOT FOUND"; \
	fi
	@printf "pip:                  "
	@if command -v pip >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1; then \
		if command -v pip3 >/dev/null 2>&1; then \
			PIP_VER=$$(pip3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1); \
		else \
			PIP_VER=$$(pip --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1); \
		fi; \
		if [ -n "$$PIP_VER" ]; then \
			PIP_MAJOR=$$(echo $$PIP_VER | cut -d'.' -f1); \
			if [ -n "$$PIP_MAJOR" ] && [ "$$PIP_MAJOR" -ge "25" ] 2>/dev/null; then \
				echo "\033[32m✓\033[0m $$PIP_VER"; \
			else \
				echo "\033[33m⚠ \033[0m $$PIP_VER (recommend $(PIP_VERSION))"; \
			fi; \
		else \
			echo "\033[33m⚠ \033[0m version unknown"; \
		fi; \
	else \
		echo "\033[31m✗\033[0m NOT FOUND"; \
	fi
	@printf "uv:                   "
	@if command -v uv >/dev/null 2>&1; then \
		UV_VER=$$(uv --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		UV_MAJOR=$$(echo $$UV_VER | cut -d'.' -f1); \
		UV_MINOR=$$(echo $$UV_VER | cut -d'.' -f2); \
		if [ "$$UV_MAJOR" -ge "1" ] || ([ "$$UV_MAJOR" = "0" ] && [ "$$UV_MINOR" -ge "4" ]); then \
			echo "\033[32m✓\033[0m $$UV_VER"; \
		else \
			echo "\033[33m⚠ \033[0m $$UV_VER (recommend >= 0.4.0)"; \
		fi; \
	else \
		echo "\033[31m✗\033[0m NOT FOUND"; \
	fi
	@printf "make:                 "
	@if command -v make >/dev/null 2>&1; then \
		MAKE_VER=$$(make --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1); \
		echo "\033[32m✓\033[0m $$MAKE_VER"; \
	else \
		echo "\033[31m✗\033[0m NOT FOUND"; \
	fi
	@printf "git:                  "
	@if command -v git >/dev/null 2>&1; then \
		GIT_VER=$$(git --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		echo "\033[32m✓\033[0m $$GIT_VER"; \
	else \
		echo "\033[31m✗\033[0m NOT FOUND"; \
	fi
	@echo ""
	@echo "Optional:"
	@echo $(DOC_DELIMITER)
	@printf "Java:                 "
	@if command -v java >/dev/null 2>&1; then \
		JAVA_VER=$$(java -version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		if [ -z "$$JAVA_VER" ]; then \
			JAVA_VER=$$(java -version 2>&1 | head -1 | grep -oE 'version "[^"]*"' | grep -oE '[0-9]+' | head -1); \
		fi; \
		echo "\033[32m✓\033[0m $$JAVA_VER (for predictive H2O-3 based methods)"; \
	else \
		echo "\033[33m⚠ \033[0m not found (for predictive H2O-3 based methods)"; \
	fi
	@printf "Graphviz:             "
	@if command -v dot >/dev/null 2>&1; then \
		DOT_VER=$$(dot -V 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		echo "\033[32m✓\033[0m $$DOT_VER (for predictive plots)"; \
	else \
		echo "\033[33m⚠ \033[0m not found (for predictive plots)"; \
	fi
	@printf "pandoc:               "
	@if command -v pandoc >/dev/null 2>&1; then \
		PANDOC_VER=$$(pandoc --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1); \
		echo "\033[32m✓\033[0m $$PANDOC_VER (for documentation)"; \
	else \
		echo "\033[33m⚠ \033[0m not found (for documentation)"; \
	fi
	@printf "gcc:                  "
	@if command -v gcc >/dev/null 2>&1; then \
		GCC_VER=$$(gcc --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		echo "\033[32m✓\033[0m $$GCC_VER (for binary distro build)"; \
	else \
		echo "\033[33m⚠ \033[0m not found (for binary distro build)"; \
	fi
	@printf "ccache:               "
	@if command -v ccache >/dev/null 2>&1; then \
		CCACHE_VER=$$(ccache --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
		echo "\033[32m✓\033[0m $$CCACHE_VER (for binary distro build)"; \
	else \
		echo "\033[33m⚠ \033[0m not found (for binary distro build)"; \
	fi
	@printf "GPU:                  "
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		CUDA_VER=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); \
		GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
		if [ -n "$$GPU_NAME" ]; then \
			echo "\033[32m✓\033[0m $$GPU_NAME (for GPU acceleration)"; \
		else \
			echo "\033[32m✓\033[0m detected (for GPU acceleration)"; \
		fi; \
	else \
		echo "\033[33m⚠ \033[0m not detected (for GPU acceleration)"; \
	fi
	@echo ""
	@echo "Environment:"
	@echo $(DOC_DELIMITER)
	@printf "Virtual env:          "
	@if [ -n "$(ACTIVE_VENV)" ]; then \
		echo "\033[32m✓\033[0m active"; \
	else \
		if [ -d ".venv" ]; then \
			echo "\033[33m⚠ \033[0m exists (not activated)"; \
		else \
			echo "\033[33m⚠ \033[0m not found (run 'make setup')"; \
		fi; \
	fi
	@printf "Git branch:           "
	@if [ -d ".git" ]; then \
		BRANCH=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null); \
		echo "\033[32m✓\033[0m $$BRANCH"; \
	else \
		echo "\033[31m✗\033[0m not a git repository"; \
	fi
	@echo ""
	@echo "Status:"
	@echo $(DOC_DELIMITER)
	@ERROR_COUNT=0; \
	WARNING_COUNT=0; \
	if ! command -v python3 >/dev/null 2>&1; then ERROR_COUNT=$$(($$ERROR_COUNT + 1)); fi; \
	if ! command -v uv >/dev/null 2>&1; then ERROR_COUNT=$$(($$ERROR_COUNT + 1)); fi; \
	if ! command -v pip >/dev/null 2>&1 && ! command -v pip3 >/dev/null 2>&1; then ERROR_COUNT=$$(($$ERROR_COUNT + 1)); fi; \
	if ! command -v make >/dev/null 2>&1; then ERROR_COUNT=$$(($$ERROR_COUNT + 1)); fi; \
	if ! command -v git >/dev/null 2>&1; then ERROR_COUNT=$$(($$ERROR_COUNT + 1)); fi; \
	if ! command -v java >/dev/null 2>&1; then WARNING_COUNT=$$(($$WARNING_COUNT + 1)); fi; \
	if ! command -v dot >/dev/null 2>&1; then WARNING_COUNT=$$(($$WARNING_COUNT + 1)); fi; \
	if ! command -v gcc >/dev/null 2>&1; then WARNING_COUNT=$$(($$WARNING_COUNT + 1)); fi; \
	if [ -z "$(ACTIVE_VENV)" ]; then WARNING_COUNT=$$(($$WARNING_COUNT + 1)); fi; \
	if [ "$$ERROR_COUNT" -gt 0 ] 2>/dev/null; then \
		echo "\033[31m✗ $$ERROR_COUNT required dependencies missing\033[0m"; \
		exit 1; \
	elif [ "$$WARNING_COUNT" -gt 0 ] 2>/dev/null; then \
		if [ "$$WARNING_COUNT" -eq 1 ]; then \
			echo "\033[33m⚠  $$WARNING_COUNT warning\033[0m"; \
		else \
			echo "\033[33m⚠  $$WARNING_COUNT warnings\033[0m"; \
		fi; \
	else \
		echo "\033[32m✓ Ready for development\033[0m"; \
	fi
	@echo ""


########
# UV CHECK
########


.PHONY: check_uv
check_uv:
ifeq ($(UV),)
	@echo "ERROR: 'uv' (Python package installer) is not installed or not on PATH."
	@echo "This project requires uv to manage Python packages."
	@echo ""
	@echo "See https://docs.astral.sh/uv/ for installation instructions."
	@exit 1
else
	@echo "OK uv is installed: $(UV)"
	@$(UV) --version
	@$(UV) --version | grep -qE 'uv [0-9]+\.([4-9]|[1-9][0-9])|uv [1-9][0-9]' || \
		(echo "WARNING: uv >= 0.4.0 recommended for full dependency-groups support" && exit 0)
endif


########
# CLEAN
########


.PHONY: documentation_clean
documentation_clean:
	@rm -rvf docs/build docs/source/notebooks docs/source/python/byoe


clean: documentation_clean
	@rm -rvf .buildinfo/
	@rm -rvf .cache
	@rm -rvf .coverage
	@rm -rvf .eggs
	@rm -rvf .pytest_cache
	@rm -rvf build
	@rm -rvf dist
	@rm -rvf h2o_sonar.egg-info
	@rm -rvf h2o_sonar/BUILD_INFO.txt
	@rm -rvf h2o-eval-studio-empty-corpus-dummy-document.txt
	@rm -rvf h2o-sonar.log
	@rm -rvf h2o-sonar/
	@rm -rvf results/
	@rm -rvf tests/build
	@rm -rvf tmp/
	@rm -vf tests/lib/given_generative_TEMPLATE.json
	@find . -type d -name "__pycache__" -exec rm -rvf {} +
	@find h2o_sonar -name "*\.c" -type f -delete


.PHONY: clean_deps
clean_deps:
	@rm -rvf deps/


.PHONY: clean_private
clean_private: clean_deps  ## clean all private data
	@rm -vf tests/lib/given_generative.json
	@rm -vf tests/lib/given_generative_TEMPLATE.json
	@rm -rvf data/generative/eval_s3/
	@rm -rvf data/generative/rag_docs/
	@echo "Private data cleaned"


.PHONY: purge
purge: clean clean_deps


.PHONY: mrproper
mrproper: purge ccache_clean
	git clean -f -d -x


.PHONY: clean_models_cache
clean_models_cache:
	mv -vf ~/.cache/huggingface ~/.cache/huggingface-$(RUN_TS)
	mv -vf ~/.cache/h2o_sonar ~/.cache/h2o_sonar-$(RUN_TS)
	mv -vf ~/nltk_data ~/nltk_data-$(RUN_TS)


.PHONY: ccache_clean
ccache_clean:
	$(C_CACHE_CLEAN)


#####################
# DOWNLOAD DATA/DEPS
#####################


# download H2O Model Validation wheel with private permissions from S3
deps/$(MV_WHEEL):
	@echo "Downloading H2O Model Validation wheel from S3 (private bucket)..."
	mkdir -p deps
	@aws s3 cp s3://h2o-sonar/dependencies/h2o-model-validation/py/$(MV_WHEEL) deps/$(MV_WHEEL) 2>/dev/null || \
	(echo "WARNING: Failed to download H2O Model Validation wheel from S3 - continuing without it..."; exit 0)


# download H2O h2oGPTe client wheel from S3 (private pre-releases) or PyPI (public releases)
ifeq ($(H2OGTPE_DISTRO_TYPE),s3)
deps/$(H2OGPTE_CLIENT_WHEEL):
	@echo "Downloading H2OGPTE client wheel from S3 (private bucket)..."
	mkdir -p deps
	aws s3 cp s3://h2o-sonar/dependencies/h2ogpte/py/$(H2OGPTE_CLIENT_WHEEL) deps/$(H2OGPTE_CLIENT_WHEEL) 2>/dev/null || \
	(echo "WARNING: Failed to download h2oGPTe client wheel from S3 - continuing without it..."; exit 0)
else
deps/$(H2OGPTE_CLIENT_WHEEL):
	mkdir -p deps
	curl -L -o deps/$(H2OGPTE_CLIENT_WHEEL) https://files.pythonhosted.org/packages/14/bc/d0d3ddd318ec8fa8c0311ceacf67d43608f4b9a876ca207a8591847d0b98/h2ogpte-1.6.50-py3-none-any.whl
endif


# SKIPPED due to H2O-3 known issue: deps/$(MV_WHEEL)
private_deps: deps/$(H2OGPTE_CLIENT_WHEEL)
	@echo "WARNING: H2O Model Validation wheel download and installation is SKIPPED because of known H2O-3 issue: H2O Model Validation wheel uses 'h2o_client' Python package which overwrites 'h2o' Python package directory when installed and thus purges H2O-3 JAR which is required for H2O Sonar to run."
	@echo "Accessible private dependencies downloaded"


# download private configuration
# - S3 is used to protect private test infrastructure / labs details
# - passwords / secrets are not a part of the configuration
# - tests/lib/given_generative_TEMPLATE.json is downloaded from S3 w/ default configuration
# - user is expected to create tests/lib/given_generative.json from the template w/ own modification
# - if user does not create tests/lib/given_generative.json, then tests/lib/given_generative_TEMPLATE.json is used (copied)
tests/lib/given_generative_TEMPLATE.json:
	@echo "Downloading private configuration..."
	aws s3 cp s3://h2o-sonar/test-config/given_generative_TEMPLATE.json $@
	@echo "DONE private configuration downloaded"


tests/lib/given_generative.json: tests/lib/given_generative_TEMPLATE.json
	@test -f $@ || cp -vf $< $@


private_test_config: tests/lib/given_generative.json
	@echo "Private configuration downloaded"


# download (big) test data from S3
# - S3 is used because of the data(set) size - data do not contain sensitive information
# - Tests which use S3 data are skipped on CI to save GHA/* traffic ~ $
.PHONY: private_test_data
private_test_data:
	@echo "Downloading private test data..."
	mkdir -p data/generative/eval_s3
	@aws s3 sync s3://h2o-sonar/test-data/ data/generative/eval_s3/ || \
	(echo "WARNING: Cannot download test data - S3 not configured - tests requiring this data will be skipped automatically."; exit 0);
	@echo "DONE sync of eval data from S3"


###############
# SETUP & BUILD
###############


.PHONY: ensure-venv
ensure-venv:
ifndef ACTIVE_VENV
	$(error Virtual environment is NOT active. Please ACTIVATE it first and make sure VIRTUAL_ENV shell variable is set.)
endif


.PHONY: venv
venv: check_uv print_make_config
	@echo "Creating virtual environment with uv..."
	rm -rf .venv
	# SEED removed as it installs pip/setuptools/wheel: $(UV) venv --seed --python $(TARGET_PYTHON_VERSION) .venv
	$(UV) venv --python $(TARGET_PYTHON_VERSION) .venv
	@echo "Virtual environment created in .venv/"
	@echo "To activate, run: source .venv/bin/activate"


setup: venv
	@echo $(DOC_DELIMITER)
	@echo "Virtual environment setup complete. To activate it, run:"
	@echo "  source ./.venv/bin/activate"


verify_version: ensure-venv
	make/verify_sonar_version.py `pwd`


build: install_setup_deps install_deps
	# Python 3.11 Ubuntu OS FIX: sudo cp /usr/include/python3.11/cpython/longintrepr.h /usr/include/python3.11/longintrepr.h
	CC="$(C_CACHE) $(CXX)" $(PYTHON) setup.py build $(DISTRO_TYPE_ARG) --cythonize


$(H2O_SONAR_WHEEL_PATH): verify_version build
	@echo "Building BINARY wheel: $(H2O_SONAR_WHEEL_PATH) ..."
	$(UV) pip install wheel
	CFLAGS="-O0" CC="$(C_CACHE) $(CXX)" $(PYTHON) setup.py build_ext $(DISTRO_TYPE_ARG) --cythonize
	CFLAGS="-O0" CC="$(C_CACHE) $(CXX)" $(PYTHON) setup.py bdist_wheel  $(DISTRO_TYPE_ARG) --cythonize -d $(DIST_DIR)
	make/fix-mac-wheel-filename.sh $(DIST_DIR)


$(H2O_SONAR_SRC_WHEEL_PATH): verify_version .buildinfo/VERSION.txt install_deps
	@echo "Building SOURCE wheel: $(H2O_SONAR_SRC_WHEEL_PATH) ..."
	$(UV) pip install wheel
	$(PYTHON) setup.py bdist_wheel $(DISTRO_TYPE_ARG) -d $(DIST_DIR)
	make/fix-mac-wheel-filename.sh $(DIST_DIR)


.PHONY: cache_models
cache_models: ensure-venv
	python -m h2o_sonar.utils.caching


############
# PACKAGING
############

dist: $(H2O_SONAR_WHEEL_PATH)
	@echo "Distribution wheel (cythonized): $(H2O_SONAR_WHEEL_PATH)"


dist_src: $(H2O_SONAR_SRC_WHEEL_PATH)
	@echo "Distribution wheel (non-cythonized): $(H2O_SONAR_SRC_WHEEL_PATH)"

.PHONY: dist_src_generative_client
dist_src_generative_client: verify_version .buildinfo/VERSION.txt install_deps_generative_client
	@echo "Building Generative AI client SOURCE wheel: $(H2O_SONAR_SRC_WHEEL_PATH) ..."
	$(UV) pip install wheel
	$(PYTHON) setup.py bdist_wheel $(DISTRO_TYPE_ARG) -d $(DIST_DIR)
	make/fix-mac-wheel-filename.sh $(DIST_DIR)


.PHONY: src_update_version
src_update_version:
	@echo "Updating H2O Sonar version in sources and documentation"
	@echo "  Last version: $(LAST_VERSION)"
	@echo "  Next version: $(NEXT_VERSION)"
	cd make && ./dist-update-release-version.sh $(LAST_VERSION) $(NEXT_VERSION) ; cd ..


tarball: ensure-venv dist documentation
	make/build-tarball.sh $(BASE_VERSION)


# all .whl to be included in the binary distribution must be prepared in dist/ directory
binary_distribution: dist documentation
	@echo "Creating binary distribution without source code, but with wheels for all supported platforms and Python versions; HTML documentation; examples and licenses"
	make/build-binary-distribution.sh $(BASE_VERSION)


gh_actions_distros: documentation_distro
	@echo "Creating distribution from GitHub Actions build artifacts provided on path: $(GH_ACTIONS_ARTIFACTS_DIR)"
	make/build-gh-actions-distribution.sh $(GH_ACTIONS_ARTIFACTS_DIR)
	mv -vf dist/h2o-sonar-$(BASE_VERSION)-documentation.zip dist/gha-distributions


sdist: ensure-venv install_deps .buildinfo/VERSION.txt
	$(PYTHON) setup.py sdist $(DISTRO_TYPE_ARG)


version: ensure-venv h2o_sonar/BUILD_INFO.txt install_setup_deps
	@$(PYTHON) setup.py $(DISTRO_TYPE_ARG) --version


.buildinfo/VERSION.txt: ensure-venv h2o_sonar/BUILD_INFO.txt install_setup_deps
	$(PYTHON) setup.py $(DISTRO_TYPE_ARG) --version > .buildinfo/VERSION.txt


.buildinfo/BUILD_INFO.txt:
	@mkdir -p .buildinfo
	@echo "build=\"$(H2O_SONAR_BUILD)\"" > $@
	@echo "suffix=\"$(H2O_SONAR_SUFFIX)\"" >> $@
	@echo "commit=\"$(H2O_SONAR_COMMIT)\"" >> $@
	@echo "branch=\"`git rev-parse HEAD | git branch -a --contains | grep -v detached | sed -e 's~remotes/origin/~~g' -e 's~^ *~~' | sort | uniq | tr '*\n' ' '`\"" >> $@
	@echo "describe=\"`git describe --always --dirty`\"" >> $@
	@echo "build_os=\"`uname -a`\"" >> $@
	@echo "build_machine=\"`hostname`\"" >> $@
	@echo "build_date=\"$(H2O_SONAR_BUILD_DATE)\"" >> $@
	@echo "build_user=\"`id -u -n`\"" >> $@
	@echo "base_version=\"$(BASE_VERSION)\"" >> $@
	@echo "version=\"{}{}\".format(base_version, suffix)" >> $@


h2o_sonar/BUILD_INFO.txt: .buildinfo/BUILD_INFO.txt
	cp .buildinfo/BUILD_INFO.txt $@


##########
# INSTALL
##########


install_analysis_deps: ensure-venv check_uv
	@echo "Installing style checking tools from pyproject.toml [dependency-groups] style..."
	# use 'uv pip install --group style' to install from dependency-groups (additive, doesn't remove other packages)
	$(UV) pip install --group style

install_setup_deps: ensure-venv check_uv
	$(UV) pip install setuptools wheel Cython==$(CYTHON_VERSION)

install_test_deps: ensure-venv check_uv
	@echo "Installing test dependencies from pyproject.toml [dependency-groups] test..."
	# use 'uv pip install --group test' to install from dependency-groups (additive, doesn't remove other packages)
	$(UV) pip install --group test

install_deps: ensure-venv check_uv h2o_sonar/BUILD_INFO.txt private_deps
	@echo "Installing all dependencies with uv..."
	$(UV) pip install -e ".[explainers,evaluators]"
	$(UV) pip install deps/$(H2OGPTE_CLIENT_WHEEL)
	# installation of H2O MV with --no-deps to avoid h2o-client/h2o conflict (bug) - it rewrites h2o/dir in the install dir
	@if [ -f deps/$(MV_WHEEL) ]; then \
		echo "Installing H2O Model Validation without dependencies to prevent h2o-client conflict..."; \
		$(UV) pip install deps/$(MV_WHEEL); \
	else \
		echo "WARNING: H2O Model Validation wheel not available - skipping installation."; \
	fi

install_deps_generative_client: ensure-venv check_uv h2o_sonar/BUILD_INFO.txt private_deps
	@echo "Installing Generative AI client dependencies with uv..."
	$(UV) pip install -e ".[genaiclient]"
	$(UV) pip install deps/$(H2OGPTE_CLIENT_WHEEL)

uninstall: ensure-venv check_uv
	$(UV) pip uninstall h2o-sonar

install: ensure-venv check_uv dist uninstall
	$(UV) pip install $(DIST_DIR)/h2o_sonar*.whl

install_src: ensure-venv check_uv dist_src uninstall
	$(UV) pip install $(DIST_DIR)/h2o_sonar*.whl


#########
# CYTHON
#########


src_to: clean install_src
	mv -v h2o_sonar ___h2o_sonar

cython_to: clean install
	mv -v h2o_sonar ___h2o_sonar

cython_from: ensure-venv check_uv
	mv -v ___h2o_sonar h2o_sonar
	make uninstall


##########
# TESTING
##########


# ALL tests: predictive AI + generative AI
test: ensure-venv install_deps install_test_deps
	rm -rf build/test-reports 2>/dev/null
	mkdir -p build/test-reports/
	$(PYTHON) -m pytest -ra --maxfail=10 $(PYTEST_FLAGS) \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
                --ignore tests/scorers \
		tests

# Cythonization test on the distribution wheel
test_dist: ensure-venv clean dist
	$(PYTHON) tests/scripts/cythonization_test.py $(DIST_DIR)


# Predictive AI smoke tests
test_predictive_smoke: ensure-venv install_deps install_test_deps
	$(PYTHON) -m pytest -m "not generative" -ra --maxfail=10 -k "not shap and not _dai" \
	tests/lib/test_container_local.py \
	tests/methods


# Predictive AI tests (skips Shapley tests as they run for a long time)
test_predictive: ensure-venv install_deps install_test_deps
	$(PYTHON) -m pytest -m "not generative" -ra --maxfail=10 $(PYTEST_FLAGS) \
        --disable-warnings \
        -k "not shap and not _dai" \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
		tests


# Predictive AI tests on cythonized (skips Shapley tests as they run for a long time)
test_predictive_cythonized: ensure-venv uninstall install cython_to install_test_deps
	$(PYTHON) -m pytest -ra --maxfail=10 $(PYTEST_FLAGS) \
        --disable-warnings \
        -k "not shap and not _dai" \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
        tests/
	mv ___h2o_sonar h2o_sonar


# Generative AI tests
test_generative: ensure-venv install_deps install_test_deps
	NUMBA_CACHE_DIR="" H2O_SONAR_CFG_DEVICE="cpu" $(PYTHON) -m pytest -m generative -ra --maxfail=1000 $(PYTEST_FLAGS) \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
		tests


# Generative AI tests on cythonized sources
test_generative_cython: ensure-venv
	@echo "BEGIN testing Generative AI on cythonized distro ..."
	$(UV) pip install $(DIST_DIR)/h2o_sonar*.whl --upgrade
	make cython_to && make test_generative
	@echo "DONE testing Generative AI on cythonized code"


# Generative AI smoke tests
test_generative_smoke: ensure-venv install_deps install_test_deps
	NUMBA_CACHE_DIR="" $(PYTHON) -m pytest -sv --maxfail=10 $(PYTEST_FLAGS) \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
		tests/lib/test_evaluate.py::test_all_evaluators


# Generative AI tests @ GPU
test_generative_gpu: ensure-venv install_deps install_test_deps
	NUMBA_CACHE_DIR="" H2O_SONAR_CFG_DEVICE="gpu" $(PYTHON) -m pytest -m generative -ra --maxfail=10 $(PYTEST_FLAGS) \
		"tests/lib/test_evaluator_answer_similarity_per_sentence.py::test_sanity" \
		"tests/lib/test_evaluator_bias_toxicity_hallucinations.py::test_sanity[h2ogpte_connection2-data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json-FairnessBiasEvaluator]" \
		"tests/lib/test_evaluator_gptscore.py::test_question_answering" \
		"tests/lib/test_evaluator_gptscore.py::test_machine_translation" \
		"tests/lib/test_evaluator_gptscore.py::test_summary_without_ref_evaluator" \
		"tests/lib/test_evaluator_gptscore.py::test_summary_with_ref_evaluator" \
		"tests/lib/test_evaluator_perplexity.py::test_evaluator" \
		"tests/lib/test_evaluator_answer_relevancy_no_judge.py::test_sanity" \
		"tests/lib/test_evaluator_context_chunk_relevancy.py::test_sanity" \
		"tests/lib/test_evaluator_groundedness.py::test_sanity" \
		"tests/lib/test_evaluator_bias_toxicity_hallucinations.py::test_sanity[h2ogpte_connection0-data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json-RagHallucinationEvaluator]" \
		"tests/lib/test_evaluator_bias_toxicity_hallucinations.py::test_sanity[h2ogpte_connection1-data/generative/toxicity_test_lab_2x_3.json-ToxicityEvaluator]" \
		"tests/lib/test_evaluator_procedure.py::test_sanity"
	@echo "SKIPPED: Kim's summarization evaluator test on GPU"


TEST_GENERATIVE_GPU_EVALUATORS="tests/lib/test_evaluator_perplexity.py::test_evaluator"
test_generative_gpu_cfg: ensure-venv install_deps install_test_deps
	@echo "Running GPU accelerated evaluator(s) w/ different CPU/GPU configs..."
	NUMBA_CACHE_DIR="" H2O_SONAR_CFG_DEVICE="" $(PYTHON) -m pytest -svv --maxfail=10 $(PYTEST_FLAGS) $(TEST_GENERATIVE_GPU_EVALUATORS)
	NUMBA_CACHE_DIR="" H2O_SONAR_CFG_DEVICE="cpu" $(PYTHON) -m pytest -svv --maxfail=10 $(PYTEST_FLAGS) $(TEST_GENERATIVE_GPU_EVALUATORS)
	NUMBA_CACHE_DIR="" H2O_SONAR_CFG_DEVICE="gpu" $(PYTHON) -m pytest -svv --maxfail=10 $(PYTEST_FLAGS) $(TEST_GENERATIVE_GPU_EVALUATORS)
	@echo "All GPU accelerated evaluator(s) test(s) passed"


TEST_DIR_GENERATIVE_AI_CLIENT := "/tmp/test-h2o-sonar-genaiclient"
test_generative_client: clean dist_src check_uv
	# SEED removed as it installs pip/setuptools/wheel: $(UV) venv --seed --python $(TARGET_PYTHON_VERSION) .venv
	@rm -rvf $(TEST_DIR_GENERATIVE_AI_CLIENT) && mkdir -v $(TEST_DIR_GENERATIVE_AI_CLIENT) && \
	cp -vf dist/$(H2O_SONAR_SRC_WHEEL_FILE) $(TEST_DIR_GENERATIVE_AI_CLIENT) && \
	cd $(TEST_DIR_GENERATIVE_AI_CLIENT) && \
	$(UV) venv --python $(TARGET_PYTHON_VERSION) .venv && \
	. ./.venv/bin/activate && \
	$(UV) pip install "$(H2O_SONAR_SRC_WHEEL_FILE)[genaiclient]" && \
	du -h | sort -h | tail && \
	echo "CHANGING DIR to $(CURRENT_DIR)" && \
	cd "$(CURRENT_DIR)" && pwd && \
	make install_deps_generative_client && \
	make install_test_deps && \
	$(UV) pip list && \
	echo "RUNNING tests/lib/test_genaiclient.py" && \
	pytest -svv --disable-warnings tests/lib/test_evaluate.py::test_all_hosts_completion


# Random attack tests
random_attack: ensure-venv
	PYTHONPATH=. $(PYTHON) tests/random_attack.py --datasets-dir=$(DATASETS_DIR) --models-index=$(MODELS_INDEX)


.PHONY: testrail
testrail:
	make/testrail.sh -f tests/build/test-reports/TEST-h2o_sonar.xml


################
# CODE ANALYSIS
################


# coverage target is used by MMC > descope it to Generative AI smoke test only
.PHONY: coverage
test-coverage: ensure-venv install_deps install_test_deps
	$(PYTHON) -m pytest $(PYTEST_FLAGS) \
		--ignore tests/scorers \
		--cov=h2o_sonar --cov-report=html:tests/build/coverage-py \
		--cov=h2o_sonar --cov-report=xml:tests/build/coverage.xml \
		--junit-prefix=$(PLATFORM) \
		--junitxml=tests/build/test-reports/TEST-h2o_sonar.xml \
		tests


code_analysis: ensure-venv install_analysis_deps
	ruff check h2o_sonar setup.py


code_analysis_mypy: ensure-venv install_analysis_deps
	mypy h2o_sonar


test_code_format_check: ensure-venv install_analysis_deps
	ruff format --check tests


test_code_format: ensure-venv install_analysis_deps
	ruff format tests


test_code_analysis: ensure-venv install_analysis_deps
	ruff check --fix tests


test_code_analysis_mypy: ensure-venv install_analysis_deps
	mypy tests


code_format: ensure-venv install_analysis_deps
	ruff format h2o_sonar setup.py
	ruff check --fix h2o_sonar setup.py


code_format_check: ensure-venv install_analysis_deps
	ruff format --check h2o_sonar setup.py


code_security: check_uv
	# pip-audit --format=html --output=tests/build/pip-audit-report.html
	# GHSA-xm59-rqc7-hhvf: nbconvert Windows-specific vuln with no fix yet (Dec 2025)
	pip-audit --ignore-vuln GHSA-xm59-rqc7-hhvf


code_compile: ensure-venv
	@echo "Compiling all source code in h2o_sonar/ to check for syntax errors..."
	find h2o_sonar -name '*.py' -type f -exec $(PYTHON) -m py_compile {} +
	@echo "Source code compilation successful"


test_compile: ensure-venv
	@echo "Compiling all test code in tests/ to check for syntax errors..."
	find tests -name '*.py' -type f -exec $(PYTHON) -m py_compile {} +
	@echo "Test code compilation successful"


lint: code_format code_analysis test_code_format test_code_analysis
	@echo ""


precommit: code_compile test_compile lint code_security
	@echo ""


code_format_analysis: code_format code_analysis test_code_format test_code_analysis


################
# DOCUMENTATION
################


.PHONY: examples
examples:
	mkdir -vp examples/predictive/byoe/examples examples/predictive/byoe/templates
	cp -vrf tests/explainers/examples/*.py examples/predictive/byoe/examples
	cp -vrf tests/explainers/templates/*.py examples/predictive/byoe/templates
	rm -vf examples/predictive/byoe/examples/__init__.py examples/predictive/byoe/examples/__init__.py


install-documentation: ensure-venv check_uv examples install_deps
	@echo "Installing documentation dependencies from pyproject.toml [dependency-groups] docs..."
	# use 'uv pip install --group docs' to install from dependency-groups (additive, doesn't remove other packages)
	$(UV) pip install --group docs


docs/source/changelog.rst: CHANGELOG.md
	pandoc --from=markdown --to=rst --output=docs/source/changelog.rst ./CHANGELOG.md
	$(PYTHON) make/fix_changelog_labels.py docs/source/changelog.rst


documentation_fast: docs/source/changelog.rst
	# prepare artifacts
	mkdir -vp docs/source/notebooks docs/source/python/byoe/examples docs/source/python/byoe/templates
	cp -vrf examples/predictive/*.ipynb docs/source/notebooks
	cp -vrf tests/explainers/examples/*.py docs/source/python/byoe/examples
	cp -vrf tests/explainers/doc/*.py docs/source/python/byoe/examples
	cp -vrf tests/explainers/templates/*.py docs/source/python/byoe/templates
	cd docs && make clean html
	@echo "Override CSS"
	cp -vf docs/source/css/theme.css docs/build/html/_static/css/theme.css
	@echo "Platform specific doc post processing ($(UNAME)) ..."
ifeq ($(PRODUCT_NAME), H2O Sonar)
	@echo "  Skipping patch as the product name is set to default"
else
ifeq ($(UNAME), Darwin)
	find docs/build/html -type f -name "*.html" -exec sed -i '' 's/H2O Sonar/$(PRODUCT_NAME)/g' {} +
else
	find docs/build/html -type f -name "*.html" -exec sed -i 's/H2O Sonar/$(PRODUCT_NAME)/g' {} +
endif
endif
	@echo "file://$(PWD)/docs/build/html/index.html"


documentation: install-documentation documentation_fast
	@echo "Documentation DONE"


documentation_es: install-documentation
	make documentation PRODUCT_NAME="H2O Eval Studio"


documentation_distro: documentation
	make/build-doc-distribution.sh $(BASE_VERSION)


documentation_zip:
	mkdir -p dist
	cd docs/build && \
	mv -vf html h2o-sonar-$(BASE_VERSION)-documentation && \
	zip -r ../../dist/h2o-sonar-$(BASE_VERSION)-documentation.zip h2o-sonar-$(BASE_VERSION)-documentation && \
	mv -vf h2o-sonar-$(BASE_VERSION)-documentation html


h2o_es_test_class_cfg: ensure-venv install_test_deps
	$(PYTHON) -m pytest -sv --disable-warnings tests/lib/test_doc_and_demos.py::test_gen_h2o_es_test_classes_config


licenses_csv: ensure-venv check_uv
	cd tests/scripts && $(PYTHON) make_licenses.py


###########
# EXAMPLES
###########


install_jupyter_deps: ensure-venv check_uv
	@echo "Installing Jupyter Lab dependencies from pyproject.toml [dependency-groups] notebooks..."
	# use 'uv pip install --group notebooks' to install from dependency-groups (additive, doesn't remove other packages)
	$(UV) pip install --group notebooks


install_h2o_sonar_for_jupyter: ensure-venv check_uv h2o_sonar/BUILD_INFO.txt
	@echo "Installing H2O Sonar in editable mode for Jupyter notebooks..."
	$(UV) pip install -e ".[explainers]"


.PHONY: examples_jupyter_lab
examples_jupyter_lab: ensure-venv install_jupyter_deps install_h2o_sonar_for_jupyter
	@echo "Registering Python kernel for Jupyter Lab..."
	$(PYTHON) -m ipykernel install --user --name=h2o-sonar --display-name="H2O Sonar"
	@echo "Starting Jupyter Lab in project root directory..."
	jupyter lab --notebook-dir=.


###############################
# EVALUATION TEST SUITE LIBRARY
# - data/generative/evals_library/h2o-eval-studio-suite-library.json
#   ... index of evaluation test suites
# - data/generative/evals_library/h2o-eval-studio-suite-library/
#   ... cached library of evaluation test suites from S3 (not in Git)
###############################

TEST_SUITE_LIBRARY_DIR = data/generative/evals_library
TEST_SUITE_LIBRARY_CACHE_DIR = $(TEST_SUITE_LIBRARY_DIR)/h2o-eval-studio-suite-library
TEST_SUITE_LIBRARY_INDEX = $(TEST_SUITE_LIBRARY_DIR)/h2o-eval-studio-suite-library.json
TEST_SUITE_LIBRARY_URLS = $(TEST_SUITE_LIBRARY_DIR)/h2o-eval-studio-suite-library-urls.txt


.PHONY: suite_lib_refresh_url_list
suite_lib_refresh_url_list:
	@echo "Refreshing the list of test case URLs to be used by wget for the library download from S3"
	cd make && \
	./test-suite-librarian.py list test-suite-urls --library-path=../$(TEST_SUITE_LIBRARY_INDEX) > ../$(TEST_SUITE_LIBRARY_URLS) \
	; cd ..


suite_lib_download: suite_lib_refresh_url_list
	mkdir -p $(TEST_SUITE_LIBRARY_CACHE_DIR)
	@echo "Downloading the latest evaluation test suite library from S3"
	wget --no-clobber -i $(TEST_SUITE_LIBRARY_URLS) -P $(TEST_SUITE_LIBRARY_CACHE_DIR)


.PHONY: suite_lib_clean
suite_lib_clean:
	rm -rf $(TEST_SUITE_LIBRARY_CACHE_DIR)


suite_lib_convert_h2o_evals: ensure-venv
	cd make && ./test-suite-librarian.py convert dir \
	--dir=../../h2o-evals \
	--output-path=../data/generative/evals_library/h2o-eval-studio-suite-library/in-gen \
	; cd ..


suite_lib_import_h2o_sonar: ensure-venv
	cd make && ./test-suite-librarian.py import dir \
	--dir=../data/generative/evals_library/h2o-eval-studio-suite-library/in-sonar \
	> ../data/generative/evals_library/h2o-eval-studio-suite-library---SONAR-generated.json && \
	./test-suite-librarian.py generate markdown \
	--library-path=../data/generative/evals_library/h2o-eval-studio-suite-library---SONAR-generated.json \
	--output-path=../data/generative/evals_library/h2o-eval-studio-suite-library---SONAR.md \
	; cd ..

suite_lib_import_moonshot: ensure-venv
	cd make && ./test-suite-librarian.py import dir \
	--dir=../data/generative/evals_library/h2o-eval-studio-suite-library/work-moonshot \
	> ../data/generative/evals_library/h2o-eval-studio-suite-library---MOONSHOT-generated.json ; cd ..


suite_lib_import_gen: ensure-venv
	cd make && ./test-suite-librarian.py import dir \
	--dir=../data/generative/evals_library/h2o-eval-studio-suite-library/in-gen \
	> ../data/generative/evals_library/h2o-eval-studio-suite-library---GEN-generated.json && \
	./test-suite-librarian.py generate markdown \
	--library-path=../data/generative/evals_library/h2o-eval-studio-suite-library---GEN-generated.json \
	--output-path=../data/generative/evals_library/h2o-eval-studio-suite-library---GEN.md \
	; cd ..


suite_lib_gen_markdown: ensure-venv
	cd make && ./test-suite-librarian.py generate markdown \
	--library-path=../data/generative/evals_library/h2o-eval-studio-suite-library.json \
	--output-path=../data/generative/evals_library/index.md ; cd ..


suite_lib_gen_html: suite_lib_gen_markdown
	cd make && ./test-suite-librarian.py generate html \
	--library-path=../data/generative/evals_library/index.md \
	--output-path=../data/generative/evals_library/index.html ; cd ..


suite_lib_gen_markdown_table: ensure-venv
	cd make && ./test-suite-librarian.py generate markdown \
	--library-path=../data/generative/evals_library/h2o-eval-studio-suite-library.json \
	--output-path=../data/generative/evals_library/h2o-eval-studio-suite-library.md \
	--table ; cd ..


##########
# RELEASE
##########


.PHONY: releng_changelog_md_new
releng_changelog_md_new:  # Generate CHANGELOG.md section for the upcoming release with the current version
	PYTHONPATH=. python make/releng-changelog-md-new.py


release_whl_src: ensure-venv dist_src
	# source wheel to dist/


release_whl_src_to_s3: ensure-venv $(H2O_SONAR_SRC_WHEEL_PATH)
	@echo "Make sure to login to AWS under the right profile first!"
	aws s3 cp $(H2O_SONAR_SRC_WHEEL_PATH) s3://eval-studio-artifacts/releases/h2o_sonar


release_whl_bin: ensure-venv dist
	# binary wheel to dist/


release_doc_zip: ensure-venv documentation_clean documentation documentation_zip
	# H2O Sonar documentation .zip to dist/


release_doc_zip_es: ensure-venv documentation_clean documentation_es documentation_zip
	# H2O Eval Studio documentation .zip to dist/


.PHONY: release_doc_to_ghpages
release_doc_to_ghpages:
	@echo "Releasing documentation to GitHub Pages branch in : $(H2O_SONAR_GH_PAGES_PATH)"
	cd $(H2O_SONAR_GH_PAGES_PATH) && \
	rm -rvf _images _sources _static css docs-licenses h2o-eval-studio notebooks *.html *.js *.inv .buildinfo && \
	echo "Ensure .nojeckyll file so that directories starting with _ are not discared by GH Actions" && \
	touch .nojekyll && \
	mkdir -vp h2o-eval-studio
	make documentation_es && cp -rvf docs/build/html/* $(H2O_SONAR_GH_PAGES_PATH)/h2o-eval-studio
	make documentation && cp -rvf docs/build/html/* $(H2O_SONAR_GH_PAGES_PATH)
	@echo "Documentation is released to GitHub Pages branch in : $(H2O_SONAR_GH_PAGES_PATH)"


release_generative: clean release_whl_src release_doc_es
	# release Generative AI artifacts


#######
# MISC
#######

.PHONY: base_version
base_version:
	@echo $(BASE_VERSION)


# refresh the build info only locally
ifeq ($(CI),)
.buildinfo/BUILD_INFO.txt: .ALWAYS_REBUILD
endif


.PHONY: ALWAYS_REBUILD
.ALWAYS_REBUILD:


.PHONY ipython:
ipython:
	uv run --with ipython ipython

# eof
