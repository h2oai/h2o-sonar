#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

GH_ACTIONS_ARTIFACTS_DIR="$1"

if [ ! -d ${GH_ACTIONS_ARTIFACTS_DIR} ]
then
    echo "Error: cannot find directory with GitHub Actions build artifacts: '${GH_ACTIONS_ARTIFACTS_DIR}'"
    exit 1
fi

SONAR_HOME=`pwd`
TARGET_DIR=${SONAR_HOME}/dist/gha-distributions

rm -rvf ${TARGET_DIR}
mkdir -vp ${TARGET_DIR}

#
# get .tgz distribution archives for all platforms
#

cd ${GH_ACTIONS_ARTIFACTS_DIR}
ls
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-macos-13-3.8-reg.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-macos-13-3.9-reg.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-centos7-3.8-dai.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-ubuntu-20.04-3.8-reg.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-ubuntu-20.04-3.9-reg.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-ubuntu-20.04-3.10-reg.zip ${TARGET_DIR}
cp -vf ${GH_ACTIONS_ARTIFACTS_DIR}/h2o-sonar-ubuntu-20.04-3.11-reg.zip ${TARGET_DIR}

cd ${TARGET_DIR}
unzip h2o-sonar-macos-13-3.8-reg.zip
unzip h2o-sonar-macos-13-3.9-reg.zip
unzip h2o-sonar-ubuntu-20.04-3.8-reg.zip
unzip h2o-sonar-ubuntu-20.04-3.9-reg.zip
unzip h2o-sonar-ubuntu-20.04-3.10-reg.zip
unzip h2o-sonar-ubuntu-20.04-3.11-reg.zip

# DAI
DAI_DIR=${TARGET_DIR}/dai
mkdir -vp ${DAI_DIR}
mv -vf h2o-sonar-centos7-3.8-dai.zip ${DAI_DIR}

rm -vf *.zip
cd ${DAI_DIR}
unzip h2o-sonar-centos7-3.8-dai.zip
rm -vf *.zip

#
# get .whl for all platforms
#

function extract_whls {
    F_TARGET_DIR=$1

    cd ${F_TARGET_DIR}
    ls *.tgz | while read F
    do
	echo "Getting .whl for $F..."
	mkdir -vp whltmp
	cp $F whltmp
	cd whltmp
	tar xf $F
	find . -name "*.whl" | while read G
	do
	    mv $G ${TARGET_DIR}
	done
	cd ..
	rm -rvf whltmp
    done
}

extract_whls ${TARGET_DIR}
extract_whls ${DAI_DIR}

# eof
