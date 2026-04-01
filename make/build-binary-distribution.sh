#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

BASE_VERSION="$1"
DIST_DIR="dist"
DOC_DIR="docs/build/html"
PYTHON=${PYTHON:-python}
PYTHON_VERSION=$(${PYTHON} -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
H2O_SONAR_DIR="h2o-sonar-${BASE_VERSION}-py${PYTHON_VERSION}-$(uname -s|tr A-Z a-z)_$(uname -m)"

if [ ! -d ${DIST_DIR} ]
then
    echo "Error: cannot create binary distribution - distribution directory doesn't exist"
    exit 1
else
    if [ ! -d ${DOC_DIR} ]
    then
        echo "Error: cannot create binary distribution - documentation build directory doesn't exist"
        exit 1
    fi
fi

function gather_files {
    mkdir -vp results doc examples/predictive/examples examples/predictive/templates licenses
    cp -v ../*.whl .
    cp -vrf ../../${DOC_DIR} doc
    cp -vrf ../../examples/*.ipynb examples
    cp -vrf ../../examples/predictive/*.ipynb examples/predictive
    cp -vrf ../../tests/explainers/examples/*.py examples/predictive/byoe/examples
    cp -vrf ../../tests/explainers/templates/*.py examples/predictive/byoe/templates
    rm -vf examples/predictive/byoe/examples/__init__.py examples/predictive/byoe/templates/__init__.py
    cp -vrf ../../licenses .
    cp -v ../../LICENSE ../../CHANGELOG.md  ../../requirements.txt .
    mkdir -vp data/predictive/models
    cp -vrf ../../data/predictive/creditcard.csv data
    cp -vrf ../../data/predictive/models/creditcard-binomial.mojo data/predictive/models
    cp -vrf ../../data/predictive/models/creditcard-binomial-sklearn-gbm.pkl data/predictive/models
    # README.md not copied intentionally - it does NOT look like OSS landing page - needs to be changed

    cd ..
    # examples should be refactored so that it can be copied and NOT cleaned (will forget a relict)
    find ./ -name "*.c" -o -name "__pycache__" -o -name "tmp" -o -name ".ipynb_checkpoints" -o -name ".gitignore" -o -name "BUILD_INFO.txt" | xargs rm -vrf

    tar zcf ${H2O_SONAR_DIR}.tgz ${H2O_SONAR_DIR}

    rm -rvf ${H2O_SONAR_DIR}
}

cd dist && rm -rvf ${H2O_SONAR_DIR} *.tgz && mkdir ${H2O_SONAR_DIR} && cd ${H2O_SONAR_DIR} && gather_files

# eof
