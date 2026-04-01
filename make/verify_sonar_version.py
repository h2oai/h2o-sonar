#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import sys


def _check_file(path: str, version: str) -> bool:
    """Verify that given file contains the expected version."""
    if not os.path.isfile(path):
        raise ValueError(f"Target file doesn't exist: '{path}'")

    version_found = False
    with open(path, "r") as file:
        data = file.readlines()
    for line in data:
        if line and version in line:
            version_found = True
            break

    return version_found


def _check_and_patch_version(git_path: str, patch: bool = False):
    """Verify H2O Sonar version, which is available in runtime, with repository version.

    Parameters
    ----------
    git_path : str
      Path to H2O Sonar Git repository.
    patch : bool
      Patch files if the version is inconsistent, fail by default.

    """
    if not os.path.isdir(git_path):
        raise ValueError(f"Git repository path doesn't exist '{git_path}'")

    version_src_path = os.path.join(git_path, "make", "version.mk")
    if not os.path.isfile(version_src_path):
        raise ValueError(f"Source file doesn't exist: '{version_src_path}'")

    with open(version_src_path, "r") as file:
        data = file.readlines()
    version = ""
    for line in data:
        if line and line.startswith("BASE_VERSION"):
            version = line[len("BASE_VERSION = ") : -1]
    if not version:
        raise RuntimeError(f"Base version not found in {version_src_path}")
    print(f"H2O Sonar version: '{version}'")

    version_py_path = os.path.join(git_path, "h2o_sonar", "version.py")
    files_to_check = [
        version_py_path,
        os.path.join(git_path, "README.md"),
    ]
    for f in files_to_check:
        if not _check_file(path=f, version=version):
            if f == version_py_path and patch:
                new_version_py_data = []
                patched = False
                for line in data:
                    if line and line.startswith("__version__"):
                        new_version_py_data.append(f'__version__ = "{version}"\n')
                        patched = True
                    else:
                        new_version_py_data.append(line)
                if not patched:
                    new_version_py_data.append(f"\n")
                    new_version_py_data.append(f'__version__ = "{version}"\n')
                with open(version_py_path, "w") as file:
                    file.writelines(new_version_py_data)
            else:
                raise RuntimeError(
                    f"Target file '{f}' must contain expected version "
                    f"specification: '{version}' "
                )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(
            f"This program must have exactly one argument with the path to H2O Sonar "
            f"Git repository (has {sys.argv[1:]} parameters)"
        )

    _check_and_patch_version(sys.argv[1])
