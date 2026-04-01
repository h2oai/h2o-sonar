#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

mkdir -vp dist
cd docs/build
pwd
ls
DOC_DIR_NAME="h2o-sonar-${1}-documentation"
mv -vf html ${DOC_DIR_NAME}
DOC_ARCHIVE_NAME="${DOC_DIR_NAME}.zip"
zip -r ${DOC_ARCHIVE_NAME} ${DOC_DIR_NAME}
mv -vf ${DOC_ARCHIVE_NAME} ../../dist

# eof
