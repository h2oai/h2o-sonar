#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Run script for H2O Sonar Generative Hello World Example
#
# This script checks for the virtual environment and runs the example.

set -e  # exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_SCRIPT="${SCRIPT_DIR}/run-example.py"

echo "========================================="
echo "H2O Sonar - Generative Hello World"
echo "========================================="

# check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Virtual environment not found at: ${VENV_DIR}"
    echo ""
    echo "Please run the installation script first:"
    echo "  ./install-example.sh"
    echo ""
    exit 1
fi

echo "OK Virtual environment found at: ${VENV_DIR}"

# check if H2O_GPTE_API_KEY is set
if [ -z "${H2O_GPTE_API_KEY}" ]; then
    echo "ERROR: H2O_GPTE_API_KEY environment variable is not set."
    echo ""
    echo "Please export your h2oGPTe API key:"
    echo "  export H2O_GPTE_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "OK H2O_GPTE_API_KEY is set"

# check if run-example.py exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

echo "OK Python script found: ${PYTHON_SCRIPT}"
echo ""

# clean up previous results
rm -rvf results *dummy-document.txt test_lab.json

# run the example using the virtual environment's Python
"${VENV_DIR}/bin/python" "${PYTHON_SCRIPT}"

echo ""
echo "========================================="
echo "OK Evaluation completed!"
echo "========================================="
echo ""
echo "Results saved to: ${SCRIPT_DIR}/results"
echo ""
echo "To view the results, open the HTML file:"
echo "  <YOUR BROWSER> results/h2o-sonar.html"
echo ""

# eof
