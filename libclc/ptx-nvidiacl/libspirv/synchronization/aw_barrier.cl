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

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierInitialize(long *state,
                                                    int expected_count) {
  if (__clc_nvvm_reflect_arch() >= 800) {
  __nvvm_mbarrier_init(state, expected_count);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierInvalidate(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    __nvvm_mbarrier_inval(state);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArrive(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_mbarrier_arrive(state);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveAndDrop(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_mbarrier_arrive_drop(state);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveNoComplete(long *state,
                                                          int count) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_mbarrier_arrive_noComplete(state, count);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF long __clc_BarrierArriveAndDropNoComplete(long *state,
                                                                 int count) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_mbarrier_arrive_drop_noComplete(state, count);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierCopyAsyncArrive(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_cp_async_mbarrier_arrive(state);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF void __clc_BarrierCopyAsyncArriveNoInc(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_cp_async_mbarrier_arrive_noinc(state);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void __clc_BarrierWait(long *state,
                                                              long arrival) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    while (!__nvvm_mbarrier_test_wait(state, arrival)) {
    }
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT bool
__clc_BarrierTestWait(long *state, long arrival) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    return __nvvm_mbarrier_test_wait(state, arrival);
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONVERGENT void
__clc_BarrierArriveAndWait(long *state) {
  if (__clc_nvvm_reflect_arch() >= 800) {
    __clc_BarrierWait(state, __clc_BarrierArrive(state));
  } else {
    __builtin_trap();
    __builtin_unreachable();
  }
}
