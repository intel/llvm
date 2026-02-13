# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# There are two modes to finding lit and filecheck
# If we are a standalone build, use the python packages
# If we are not a standalone build, use the one provided by LLVM
# Note that the python filecheck re-implementation isn't 100% compatible, but hopefully for simple tests it should be
# good enough

if(UR_STANDALONE_BUILD)
  # Standalone build
  find_program(URLIT_LIT_BINARY "lit")
  if(NOT URLIT_LIT_BINARY)
    message(FATAL_ERROR "No `lit` binary was found in path. Consider installing the python Package `lit`")
  endif()

  find_program(URLIT_FILECHECK_BINARY NAMES "filecheck" "FileCheck")
  if(NOT URLIT_FILECHECK_BINARY)
    message(FATAL_ERROR "No `filecheck` or `FileCheck` binary was found in path. Consider installing the python Package `filecheck`")
  endif()
else()
  # In source build

  # This will be in the $PATH
  set(URLIT_FILECHECK_BINARY "FileCheck")
endif()
