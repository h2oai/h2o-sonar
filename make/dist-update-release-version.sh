#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# Usage: make/dist-update-release-version.sh <old-version> <new-version>
#
# Examples:
#
#   dist-update-release-version.sh 2.4.0 2.5.0
#   dist-update-release-version.sh 1.3.0-rc31 1.3.0-rc32
#   dist-update-release-version.sh 1.3.0rc31 1.3.0rc32
#
# Hints:
#
# - two runs are typically needed for RCs - 1.3.1-rc1 and 1.3.1rc1
#

# check that old-version and new-version are specified
if [ "$#" -ne 2 ]; then
    echo "Error: old-version and new-version must be specified e.g. 1.3.1 1.2.0rc3"
    exit 1
fi

# update make/version.mk
sed -i "s/${1}/${2}/g" ../make/version.mk

# update h2o_sonar/version.py
sed -i "s/${1}/${2}/g" ../h2o_sonar/version.py

# update H2O Sonar version in README.md
sed -i "s/${1}/${2}/g" ../README.md

# update version in the pyproject.toml
sed -i "s/${1}/${2}/g" ../pyproject.toml

# eof
