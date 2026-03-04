//===--------------------- usm.hpp - OpenCL Adapter -----------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/cl_ext.h"
#include <CL/cl.h>

// This struct is intended to be used in conjunction with the below callback via
// clSetEventCallback to release temporary allocations created by the adapter to
// implement certain USM operations.
//
// Example usage:
//
// auto Info = new AllocDeleterCallbackInfoUSM(USMFreeFuncPtr, Context,
// Allocation); clSetEventCallback(USMOpEvent, CL_COMPLETE,
// AllocDeleterCallback, Info);

struct AllocDeleterCallbackInfoBase {
  AllocDeleterCallbackInfoBase(cl_context Context, uint8_t *Allocation)
      : CLContext(Context), Allocation(Allocation) {
    clRetainContext(CLContext);
  }

  virtual ~AllocDeleterCallbackInfoBase() { clReleaseContext(CLContext); }

  AllocDeleterCallbackInfoBase(const AllocDeleterCallbackInfoBase &) = delete;
  AllocDeleterCallbackInfoBase &
  operator=(const AllocDeleterCallbackInfoBase &) = delete;

protected:
  cl_context CLContext;
  uint8_t *Allocation;
};

struct AllocDeleterCallbackInfo : AllocDeleterCallbackInfoBase {
  AllocDeleterCallbackInfo(cl_context CLContext, uint8_t *Allocation)
      : AllocDeleterCallbackInfoBase(CLContext, Allocation) {}

  ~AllocDeleterCallbackInfo() override { delete[] Allocation; }
};

struct AllocDeleterCallbackInfoUSM : AllocDeleterCallbackInfoBase {
  AllocDeleterCallbackInfoUSM(clMemBlockingFreeINTEL_fn USMFree,
                              cl_context CLContext, uint8_t *Allocation)
      : AllocDeleterCallbackInfoBase(CLContext, Allocation), USMFree(USMFree) {}
  ~AllocDeleterCallbackInfoUSM() override { USMFree(CLContext, Allocation); }

  clMemBlockingFreeINTEL_fn USMFree;
};

struct AllocDeleterCallbackInfoUSMWithQueue : AllocDeleterCallbackInfoUSM {
  AllocDeleterCallbackInfoUSMWithQueue(clMemBlockingFreeINTEL_fn USMFree,
                                       cl_context CLContext,
                                       uint8_t *Allocation,
                                       cl_command_queue CLQueue)
      : AllocDeleterCallbackInfoUSM(USMFree, CLContext, Allocation),
        CLQueue(CLQueue) {}
  ~AllocDeleterCallbackInfoUSMWithQueue() override {
    clReleaseCommandQueue(CLQueue);
  }

  cl_command_queue CLQueue;
};

template <class T>
void AllocDeleterCallback(cl_event event, cl_int, uint8_t *pUserData);
