#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Install script for H2O Sonar Generative CLI Hello World Example
#
# This script:
# 1. Checks for h2oGPTe API key
# 2. Creates a Python 3.11 virtual environment (if not present)
# 3. Installs h2o-sonar from a wheel file in the current directory

set -e  # exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "========================================="
echo "H2O Sonar - Generative Example"
echo "========================================="

# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' command not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# check if H2O_GPTE_API_KEY is set
if [ -z "${H2O_GPTE_API_KEY}" ]; then
    echo "ERROR: H2O_GPTE_API_KEY environment variable is not set."
    echo ""
    echo "This example requires h2oGPTe access. To get your API key:"
    echo "  1. Sign up for h2oGPTe: https://h2ogpte.h2o.ai/"
    echo "  2. Navigate to Settings > API Keys"
    echo "  3. Create a new API key"
    echo "  4. Export it: export H2O_GPTE_API_KEY='your-api-key-here'"
    echo ""
    echo "Alternatively, for testing you can use H2O AI Cloud:"
    echo "  - https://h2o.ai/"
    echo ""
    exit 1
fi

echo "OK H2O_GPTE_API_KEY environment variable is set"

# check if virtual environment exists
if [ -d "${VENV_DIR}" ]; then
    echo "OK Virtual environment found at: ${VENV_DIR}"
else
    echo "Creating virtual environment with Python 3.11..."
    uv venv "${VENV_DIR}" --python 3.11
    echo "OK Virtual environment created at: ${VENV_DIR}"
fi

# check if h2o-sonar wheel exists in current directory
WHEEL_FILE=$(ls "${SCRIPT_DIR}"/h2o_sonar*.whl 2>/dev/null | head -n 1)

if [ -z "${WHEEL_FILE}" ]; then
    echo "ERROR: h2o-sonar wheel file (h2o_sonar*.whl) not found in ${SCRIPT_DIR}"
    echo ""
    echo "Please get h2o-sonar wheel file to this directory:"
    echo "  https://github.com/h2oai/h2o-sonar#installation"
    echo ""
    exit 1
fi

echo "OK Found h2o-sonar wheel: $(basename ${WHEEL_FILE})"

# uninstall any existing h2o-sonar installation
echo "Uninstalling existing h2o-sonar (if present)..."
uv pip uninstall -y h2o-sonar --python "${VENV_DIR}/bin/python" 2>/dev/null || true

# install h2o-sonar with evaluators extras
echo "Installing h2o-sonar from wheel..."
uv pip install "${WHEEL_FILE}[evaluators]" --python "${VENV_DIR}/bin/python"
echo "OK h2o-sonar installed"

# install h2ogpte client from PyPI
echo "Installing h2ogpte client from PyPI..."
uv pip install h2ogpte --python "${VENV_DIR}/bin/python"
echo "OK h2ogpte client installed"

echo "========================================="
echo "OK Installation completed successfully!"
echo "========================================="
echo ""
echo "To run the example:"
echo "  ./run-example.sh"
echo ""

# eof
