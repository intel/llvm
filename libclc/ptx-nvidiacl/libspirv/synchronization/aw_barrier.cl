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

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveAndDrop(long* state) {
  return __nvvm_mbarrier_arrive_drop(state);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveNoComplete(long* state, int count) {
  return __nvvm_mbarrier_arrive_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF long __spirv_BarrierArriveAndDropNoComplete(long* state, int count) {
  return __nvvm_mbarrier_arrive_drop_noComplete(state, count);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierCopyAsyncArrive(long* state) {
  return __nvvm_cp_async_mbarrier_arrive(state);
}

_CLC_OVERLOAD _CLC_DEF void __spirv_BarrierCopyAsyncArriveNoInc(long* state) {
  return __nvvm_cp_async_mbarrier_arrive_noinc(state);
}

_CLC_OVERLOAD _CLC_DEF int __spirv_BarrierPendingCount(long arrival) {
  return __nvvm_mbarrier_pending_count(arrival);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __spirv_BarrierWait(long* state, long arrival) {
  while(!__nvvm_mbarrier_test_wait(state, arrival)){}
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT bool __spirv_BarrierTestWait(long* state, long arrival) {
  return __nvvm_mbarrier_test_wait(state, arrival);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __spirv_BarrierArriveAndWait(long* state) {
  __spirv_BarrierWait(state, __spirv_BarrierArrive(state));
}