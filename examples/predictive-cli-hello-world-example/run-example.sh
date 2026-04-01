#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Run script for H2O Sonar Predictive CLI Hello World Example
#

set -e  # exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "========================================="
echo "H2O Sonar - Predictive CLI Hello World"
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
echo ""

# clean up previous results
rm -rvf results

# run h2o-sonar CLI from the virtual environment w/ DEFAULT evaluators
"${VENV_DIR}/bin/h2o-sonar" run interpretation \
    --dataset creditcard.csv \
    --model creditcard-binomial-sklearn-1.8.0-gbm.pkl \
    --target-col "default payment next month" \
    --results-location results

echo ""
echo "========================================="
echo "OK Interpretation completed!"
echo "========================================="
echo ""
echo "Results saved to: ${SCRIPT_DIR}/results"
echo ""
echo "To view the results, open the HTML files:"
echo "  <YOUR BROWSER> results/h2o-sonar.html"
echo ""

# eof
