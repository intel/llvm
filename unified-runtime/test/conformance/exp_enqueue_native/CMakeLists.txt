# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (UR_BUILD_ADAPTER_CUDA)
  add_conformance_test_with_kernels_environment(
    exp_enqueue_native
    enqueue_native_cuda.cpp
  )
  target_include_directories(test-exp_enqueue_native PRIVATE
      ${PROJECT_SOURCE_DIR}/source
      ${PROJECT_SOURCE_DIR}/source/adapters/cuda
  )
  target_link_libraries(test-exp_enqueue_native PRIVATE cudadrv)
endif()

# TODO: Add more tests for different triples
