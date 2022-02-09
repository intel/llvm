//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>

int __nvvm_reflect(const char __constant *);

_CLC_OVERLOAD _CLC_DEF void __spirv_GroupWaitEvents(unsigned int scope,
                                                    int num_events,
                                                    event_t *event_list) {
  if (__nvvm_reflect("__CUDA_ARCH") >= 800) {
    __nvvm_cp_async_wait_all();
  }
  __spirv_ControlBarrier(scope, scope, SequentiallyConsistent);
}
