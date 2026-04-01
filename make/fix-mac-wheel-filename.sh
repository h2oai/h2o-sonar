#!/bin/bash
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.

set -e

function fix_mac_wheel_filename() {
  filename="${1:-}"
  if [[ -f "$filename" ]] && [[ ${filename} =~ (h2o.*macosx_)10_[0-9]?[0-9](.*.whl) ]]
  then
    dest_name="${BASH_REMATCH[1]}11_0${BASH_REMATCH[2]}"
    echo "Renaming ${filename} to ${dest_name}"
    mv "${filename}" "${dest_name}"
  fi
}


dist_dir=${1:-}
if [[ ! -d "${dist_dir}" ]]
then
  echo "Usage ${0} <dist directory>"
  exit 1
fi

cd ${dist_dir}
dist_file=$(ls -t h2o_sonar*.whl|head -1)
fix_mac_wheel_filename ${dist_file}
