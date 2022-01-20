//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierInitializeINTEL(long *state,
                                                      int expected_count) {
  __nvvm_mbarrier_init(state, expected_count);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierInvalidateINTEL(long *state) {
  __nvvm_mbarrier_inval(state);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveINTEL(long *state) {
  return __nvvm_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveAndDropINTEL(long *state) {
  return __nvvm_mbarrier_arrive_drop(state);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveNoCompleteINTEL(long *state,
                                                            int count) {
  return __nvvm_mbarrier_arrive_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveAndDropNoCompleteINTEL(long *state,
                                                                   int count) {
  return __nvvm_mbarrier_arrive_drop_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierCopyAsyncArriveINTEL(long *state) {
  return __nvvm_cp_async_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierCopyAsyncArriveNoIncINTEL(long *state) {
  return __nvvm_cp_async_mbarrier_arrive_noinc(state);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __spirv_BarrierWaitINTEL(long *state,
                                                                long arrival) {
  while (!__nvvm_mbarrier_test_wait(state, arrival)) {
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT bool
__spirv_BarrierTestWaitINTEL(long *state, long arrival) {
  return __nvvm_mbarrier_test_wait(state, arrival);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__spirv_BarrierArriveAndWaitINTEL(long *state) {
  __spirv_BarrierWaitINTEL(state, __spirv_BarrierArriveINTEL(state));
}