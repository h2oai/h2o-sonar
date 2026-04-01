#!/usr/bin/env python
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

import importlib.resources
import os


__all__ = ["__version__", "__build_info__"]

# Build defaults
build_info = {
    "suffix": "+local",
    "build": "dev",
    "commit": "",
    "describe": "",
    "build_os": "",
    "build_machine": "",
    "build_date": "",
    "build_user": "",
    "base_version": "0.0.0",
}

path = str(importlib.resources.files("h2o_sonar") / "BUILD_INFO.txt")
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        exec(f.read(), build_info)

# Exported properties
__version__ = f"{build_info['base_version']}{build_info['suffix']}"
__build_info__ = build_info
