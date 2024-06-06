# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is lifted from intel-llvm's FindCUDACupti implementation
# https://github.com/intel/llvm/blob/0cd04144d9ca83371c212e8e4709a59c968291b9/sycl/cmake/modules/FindCUDACupti.cmake

macro(find_cuda_cupti_library)
  find_library(CUDA_cupti_LIBRARY
    NAMES cupti
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
          ENV CUDA_PATH
    PATH_SUFFIXES nvidia/current lib64 lib/x64 lib
                  ../extras/CUPTI/lib64/
                  ../extras/CUPTI/lib/
  )
endmacro()

macro(find_cuda_cupti_include_dir)
  find_path(CUDA_CUPTI_INCLUDE_DIR cupti.h PATHS
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}/../extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}"
      NO_DEFAULT_PATH)
endmacro()
