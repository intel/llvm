#!/bin/bash
# Copyright (c) 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Script to determine if source code in Pull Request is properly formatted.
# Exits with non 0 exit code if formatting is needed.
#
# This script assumes to be invoked at the project root directory.

# Changes made to the initial version from
# https://github.com/KhronosGroup/SPIRV-Tools:
# * Renamed FILES_TO_CHECK to MODIFIED_FILES;
# * Add a filtering step on the files to check, with the result stored in
#   FILES_TO_CHECK.

if [ -n "${TRAVIS_BRANCH+check}" ]; then
    BASE_BRANCH="travis/${TRAVIS_BRANCH}"
    git fetch --depth=1 origin ${TRAVIS_BRANCH}:${BASE_BRANCH}
else
    BASE_BRANCH=$(git for-each-ref --format='%(upstream:short)' "$(git symbolic-ref -q HEAD)")
fi

if [ -z ${BASE_BRANCH} ]; then
    cat <<EOF
Error: no branch to compare with.
If you are running script locally, please set upstream branch
that you want to compare your changes with: "git branch --set-upstream-to=<branch>".
EOF
    exit 2
fi

echo "Checking changes between '${BASE_BRANCH}' and 'HEAD'..."

MODIFIED_FILES=$(git diff --name-only ${BASE_BRANCH} | grep -E ".*\.(cpp|cc|c\+\+|cxx|c|h|hpp)$")
FILES_TO_CHECK=$(echo "${MODIFIED_FILES}" | grep -v -E "Mangler/*|runtime/*|libSPIRV/(OpenCL.std.h|spirv.hpp)$")

if [ -z "${FILES_TO_CHECK}" ]; then
  echo "No source code to check for formatting."
  exit 0
fi

FORMAT_DIFF=$(git diff -U0 ${BASE_BRANCH} -- ${FILES_TO_CHECK} | ./utils/clang-format-diff.py -p1 -style=file)

if [ -z "${FORMAT_DIFF}" ]; then
  echo "All source code in PR properly formatted."
  exit 0
else
  echo "Found formatting errors!"
  echo "${FORMAT_DIFF}"
  exit 1
fi
