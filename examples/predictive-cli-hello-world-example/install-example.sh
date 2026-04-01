#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Install script for H2O Sonar Predictive Hello World Example
#

set -e  # exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "========================================="
echo "H2O Sonar - Example"
echo "========================================="

# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' command not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# check if java is installed (required for H2O-3 backend)
if ! command -v java &> /dev/null; then
    echo "ERROR: 'java' command not found. Java 1.7+ is required for H2O-3 backend."
    echo ""
    echo "Please install Java first:"
    echo "  - On Ubuntu/Debian: sudo apt-get install default-jdk"
    echo "  - Or download from: https://www.oracle.com/java/technologies/downloads/"
    echo ""
    exit 1
fi

# verify java version
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}')
echo "OK Java found: ${JAVA_VERSION}"

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
# install h2o-sonar with explainers extras
echo "Installing h2o-sonar from wheel..."
uv pip install "${WHEEL_FILE}[explainers]" --python "${VENV_DIR}/bin/python"
echo "OK h2o-sonar installed"

echo "========================================="
echo "OK Installation completed successfully!"
echo "========================================="
echo ""
echo "To run the example:"
echo "  ./run-example.sh"
echo ""

# eof
