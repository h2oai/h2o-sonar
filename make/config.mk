# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

###############################
# BUILD CONFIGURATION VARIABLES
###############################

BUILD_DIR := build
RUN_TS   := $(shell date +%Y-%m-%d--%H-%M-%S)
PYTHON    ?= python
MODULE    ?= .
NPROCS    := $(shell nproc)

ifneq ($(CI),)
PYTEST_FLAGS := -vv -s --tb=native
endif

##################
# Platform details
##################

OS       := $(shell uname | tr A-Z a-z)
ARCH     := $(shell uname -m)
PLATFORM := $(ARCH)-$(OS)

######################
# Distribution details
######################

DISTRO_TYPE_ARG := --regular

H2O_MLI_VERSION                  ?= 1.10.29
H2O_WHEEL_VERSION                ?= 3.46.0.4

DATATABLE_VERSION                ?= 1.1.0a2234
MOJO2_CPP_RUNTIME_VERSION        ?= 2.7.11
MV_VERSION                       ?=0.16.3

################
# Python version
################

PYTHON ?= python
PYTHON_VERSION_FULL := $(wordlist 2,4,$(subst ., ,$(shell $(PYTHON) --version 2>&1)))
PYTHON_VERSION_MAJOR := $(word 1,${PYTHON_VERSION_FULL})
PYTHON_VERSION_MINOR := $(word 2,${PYTHON_VERSION_FULL})
PYTHON_VERSION_PATCH := $(word 3,${PYTHON_VERSION_FULL})
PYTHON_CP_VERSION := ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}

#####################################
# Python version driven configuration
#####################################

TARGET_PYTHON_VERSION := 3.11

PIP_VERSION = 25.3.0
CYTHON_VERSION := 3.1.4  # 3.8 version: 0.29.37
MOJO2_CPP_RUNTIME_VERSION := 2.8.2

WHL_CP_SECTION := cp$(PYTHON_CP_VERSION)-cp$(PYTHON_CP_VERSION)

########################
# Distribution directory
########################

DIST_DIR := dist/

# cythonized wheel
H2O_SONAR_WHEEL_FILE := h2o_sonar-$(BASE_VERSION)-$(WHL_CP_SECTION)-$(OS)_$(ARCH).whl
H2O_SONAR_WHEEL_PATH := $(DIST_DIR)$(H2O_SONAR_WHEEL_FILE)

# source wheel: OS=none, Platform=any ~ h2o_sonar-1.0.0-py3-none-any.whl
H2O_SONAR_SRC_WHEEL_FILE := h2o_sonar-$(BASE_VERSION)-py3-none-any.whl
H2O_SONAR_SRC_WHEEL_PATH := $(DIST_DIR)$(H2O_SONAR_SRC_WHEEL_FILE)

###########
# C COMPILE
###########

CXX ?= gcc
C_CACHE=$(shell echo `which ccache`)
ifeq ($(C_CACHE),)
	C_CACHE_CLEAN=
else
	C_CACHE_CLEAN=$(C_CACHE) -C
endif

HTTP_ARTIFACTS_BUCKET := https://s3.amazonaws.com/artifacts.h2o.ai

################
# Util functions
################

# URL encode
urlenc = $(subst +,%2B,$1)

# Version to bucket name
# For example, version 1.2.3 -> "$(HTTP_ARTIFACTS_BUCKET)/releases", version 1.2.3-SNAPSHOT -> "$(HTTP_ARTIFACTS_BUCKET)/snapshots"
version2bucket = $(HTTP_ARTIFACTS_BUCKET)/$(if $(findstring SNAPSHOT, $1),snapshots,releases)

##############
# DEPENDENCIES
##############

#
# h2oGPTe client
#
H2OGPTE_CLIENT_WHEEL = h2ogpte-1.6.50-py3-none-any.whl

#
# Datatable
#
DATATABLE_GAV                     = https://s3.amazonaws.com/h2o-release/datatable/dev/datatable-$(DATATABLE_VERSION)
# Dummy mapping to modify version based on OS and ARCH
DATATABLE_MANYLINUX_linux_x86_64  = manylinux_2_17_x86_64
DATATABLE_MANYLINUX_linux_ppc64le = manylinux2014_ppc64le
DATATABLE_MANYLINUX_darwin_x86_64 = macosx_11_0_x86_64
DATATABLE_WHEEL                   = datatable-$(DATATABLE_VERSION)-$(WHL_CP_SECTION)-$(DATATABLE_MANYLINUX_$(OS)_$(ARCH)).whl
DATATABLE_WHEEL_URL               = datatable@$(call urlenc,$(DATATABLE_GAV)/$(DATATABLE_WHEEL))

#
# H2O Model Validation
#
MV_WHEEL                        = h2o_mv-$(MV_VERSION)-py3-none-any.whl
