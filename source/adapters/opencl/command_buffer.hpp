//===--------- command_buffer.hpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl_ext.h>
#include <ur/ur.hpp>

struct ur_exp_command_buffer_handle_t_ {
  ur_queue_handle_t hInternalQueue;
  ur_context_handle_t hContext;
  cl_command_buffer_khr CLCommandBuffer;

  ur_exp_command_buffer_handle_t_(ur_queue_handle_t hQueue,
                                  ur_context_handle_t hContext,
                                  cl_command_buffer_khr CLCommandBuffer)
      : hInternalQueue(hQueue), hContext(hContext),
        CLCommandBuffer(CLCommandBuffer) {}
};
