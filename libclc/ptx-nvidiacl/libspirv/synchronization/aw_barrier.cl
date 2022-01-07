//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierInitialize(long* state,
                                                        int expected_count) {
  __nvvm_mbarrier_init(state, expected_count);
}

_CLC_OVERLOAD _CLC_DEF void
__spirv_BarrierInvalidate(long* state) {
  __nvvm_mbarrier_inval(state);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArrive(long* state) {
  return __nvvm_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __spirv_BarrierWait(long* state, long arrival) {
  while(!__nvvm_mbarrier_test_wait(state, arrival)){}
}