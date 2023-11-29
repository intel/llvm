// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <hip/hip_runtime.h>

namespace pi {

// utility function to clear the HIP context stack
inline void clearHipContext() {
  hipCtx_t ctxt = nullptr;
  do {
    hipCtxSetCurrent(nullptr);
    hipCtxGetCurrent(&ctxt);
  } while (ctxt != nullptr);
}

} // namespace pi
