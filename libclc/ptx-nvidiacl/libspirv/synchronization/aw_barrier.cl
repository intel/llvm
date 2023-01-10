//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

extern int __clc_nvvm_reflect_arch();

void __clc_trap_if_sm_lower_than_80() {
  if (__clc_nvvm_reflect_arch() < 800) {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierInitialize(long *state,
                                                    int expected_count) {
  __clc_trap_if_sm_lower_than_80();
  __nvvm_mbarrier_init(state, expected_count);
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierInvalidate(long *state) {
  __clc_trap_if_sm_lower_than_80();
  __nvvm_mbarrier_inval(state);
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArrive(long *state) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveAndDrop(long *state) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_mbarrier_arrive_drop(state);
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveNoComplete(long *state,
                                                          int count) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_mbarrier_arrive_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveAndDropNoComplete(long *state,
                                                                 int count) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_mbarrier_arrive_drop_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierCopyAsyncArrive(long *state) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_cp_async_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierCopyAsyncArriveNoInc(long *state) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_cp_async_mbarrier_arrive_noinc(state);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __clc_BarrierWait(long *state,
                                                              long arrival) {
  __clc_trap_if_sm_lower_than_80();
  while (!__nvvm_mbarrier_test_wait(state, arrival)) {
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT bool
__clc_BarrierTestWait(long *state, long arrival) {
  __clc_trap_if_sm_lower_than_80();
  return __nvvm_mbarrier_test_wait(state, arrival);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__clc_BarrierArriveAndWait(long *state) {
  __clc_BarrierWait(state, __clc_BarrierArrive(state));
}
