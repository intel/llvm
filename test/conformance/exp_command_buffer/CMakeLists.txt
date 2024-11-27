# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_kernels_environment(exp_command_buffer
  release.cpp
  retain.cpp
  commands.cpp
  fill.cpp
  event_sync.cpp
  kernel_event_sync.cpp
  update/buffer_fill_kernel_update.cpp
  update/invalid_update.cpp
  update/kernel_handle_update.cpp
  update/usm_fill_kernel_update.cpp
  update/buffer_saxpy_kernel_update.cpp
  update/ndrange_update.cpp
  update/usm_saxpy_kernel_update.cpp
  update/event_sync.cpp
  update/kernel_event_sync.cpp
  update/local_memory_update.cpp
)
