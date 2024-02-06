// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda.h>

namespace pi {

// utility function to clear the CUDA context stack
inline void clearCudaContext() {
  CUcontext ctxt = nullptr;
  do {
    cuCtxSetCurrent(nullptr);
    cuCtxGetCurrent(&ctxt);
  } while (ctxt != nullptr);
}

} // namespace pi
