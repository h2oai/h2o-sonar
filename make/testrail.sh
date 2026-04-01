#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
# H2O-Sonar Testrail Integration

# Terminate the script if any command fails
set -e

# Display help information
helpFunction(){
    echo "H2O-Sonar Testrail Integration"
    echo "-------------------------------"
    echo
    echo "Usage: ./testrail.sh -f <xml_file>"
    echo "Available options:"
    echo "-f    Specify the JUnit XML file containing test results from a pytest run."
    echo "-h    Display this help message."
    echo 
    exit 1 # Exit script after printing help
}

# Display error message and exit
errorExit() {
    echo "Error: $1" >&2
    exit 0
}

# Check if the Python version is compatible
checkPythonVersion() {
    PY_VERSION=$(python --version)
    if [[ ${PY_VERSION} == *"Python 3.9"* ]]; then
        echo -e "The script can be executed using ${PY_VERSION}."
    else
        errorExit "The script is programmed to execute with Python 3.9, but the server is running on ${PY_VERSION}."
    fi
}

# If no arguments provided, display help and exit
if [ "$#" -eq 0 ]
then
    helpFunction
    exit 1
fi

# Parse command-line options
while getopts f:h: flag
do
    case "${flag}" in
        f) FILE=${OPTARG};;
        h) helpFunction ;;
        ?) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

# Script Variables
GIT_HASH=$(git rev-parse --short HEAD)

# Check if the target branch is 'dev-*'
if [[ "${TARGET_BRANCH}" == *"dev"* ]]
then
    echo -e "... '${FEATURE_BRANCH}' is merging into '${TARGET_BRANCH}' ..."
    echo -e "Git branch '${FEATURE_BRANCH}' qualifies for Testrail Integration"
else
    # Exit script for all other branches
    echo -e "...'${FEATURE_BRANCH}' is merging into '${TARGET_BRANCH}'..."
    echo -e "Git branch '${FEATURE_BRANCH}' is NOT eligible for Testrail Integration"
    echo -e "... Exiting ..."
    exit 0
fi

# Check Python version compatibility
checkPythonVersion
echo -e "... Uploading results to TestRail..."

TITLE="${GIT_HASH}_${FEATURE_BRANCH}"
if [ ${#TITLE} -gt 250 ]; then
    TITLE="${TITLE:0:250}"
fi

# Upload results to TestRail using trcli
trcli --yes \
    --host https://h2o.testrail.io/ \
    --project "H2O-Sonar" \
    --username ${TESTRAIL_USER} \
    --password ${TESTRAIL_PASS} \
    parse_junit \
    --title "${GIT_HASH}_${FEATURE_BRANCH}" \
    --file ${FILE} \
    --suite-id 349

# eof
