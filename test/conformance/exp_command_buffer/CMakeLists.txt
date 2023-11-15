# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_kernels_environment(exp_command_buffer
  buffer_fill_kernel_update.cpp
  usm_fill_kernel_update.cpp
  buffer_saxpy_kernel_update.cpp
  usm_saxpy_kernel_update.cpp
  ndrange_update.cpp
  release.cpp
  retain.cpp
  invalid_update.cpp
)
