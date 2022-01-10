//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

_CLC_OVERLOAD _CLC_DEF void __spirv_MemoryBarrier(unsigned int memory,
                                                  unsigned int semantics) {
  __nvvm_membar_cta();
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_ControlBarrier(unsigned int scope, unsigned int memory,
                       unsigned int semantics) {
  if (scope == Subgroup) {
    // use a full mask as barriers are required to be convergent and exited
    // threads can safely be in the mask
    __nvvm_bar_warp_sync(0xFFFFFFFF);
  } else {
    __syncthreads();
  }
}
