#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

if [ "$#" -ne 3 ]; then
    echo "Error: script must get 4 parameters: DAI version, Python, platform and PyTest flags"
    exit 1
fi

export DAI_VERSION=${1}
export PYTHON=${2}
export PLATFORM=${3}
export PYTEST_FLAGS="${4}"
export VENV_NAME=".venv-dai-${DAI_VERSION}"

echo "========================================================================="
echo "Running DAI tests w/ '$DAI_VERSION' '$PYTHON' '$PLATFORM'" '$PYTEST_FLAGS'

function runTests {    
    . ${VENV_NAME}/bin/activate

    make install_deps
    make install_test_deps
    rm -rf build/test-dai-scoring-reports 2>/dev/null
    mkdir -p build/test-dai-scoring-reports/
    ${PYTHON} -m pytest -s -v -ra --maxfail=10 ${PYTEST_FLAGS} --junit-prefix=${PLATFORM} --junitxml=tests/build/test-dai-scoring-reports/TEST-h2o_sonar.xml tests/dai_scoring
}

cd ../h2o-sonar && runTests

# eof
