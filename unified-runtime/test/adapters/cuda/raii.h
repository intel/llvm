// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_TEST_CONFORMANCE_ADAPTERS_CUDA_RAII_H_INCLUDED
#define UR_TEST_CONFORMANCE_ADAPTERS_CUDA_RAII_H_INCLUDED

#include "uur/raii.h"
#include <cuda.h>

struct RAIICUevent {
  CUevent handle = nullptr;

  ~RAIICUevent() {
    if (handle) {
      cuEventDestroy(handle);
    }
  }

  CUevent *ptr() { return &handle; }
  CUevent get() { return handle; }
};

#endif // UR_TEST_CONFORMANCE_ADAPTERS_CUDA_RAII_H_INCLUDED
