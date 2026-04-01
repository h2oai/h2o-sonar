# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import sys
from zipfile import ZipFile


python_extenstion = [".py", ".pyc"]


def main(dist_dir):
    if not os.path.exists(dist_dir) or not os.path.isdir(dist_dir):
        raise ValueError("Directory %s does not exist!" % dist_dir)

    wheels_to_check = list(filter(lambda file: ".whl" in file, os.listdir(dist_dir)))

    for wheel in wheels_to_check:
        relative_wheel_path = os.path.join(dist_dir, wheel)
        print("Checking file %s for unobfuscated Python files..." % relative_wheel_path)
        files_in_wheel = ZipFile(relative_wheel_path).namelist()
        for file_in_wheel in files_in_wheel:
            if python_file(file_in_wheel):
                raise ValueError(
                    "Wheel file %s contains .py/.pyc file(s) - %s!"
                    % (wheel, file_in_wheel)
                )


def python_file(file_name):
    return any(file_name.endswith(extenstion) for extenstion in python_extenstion)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Only the distribution directory should be passed.")

    main(sys.argv[1])
