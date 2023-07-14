//==------- native_cpu.hpp - Native CPU helper header ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "cg_types.hpp"
#include <CL/__spirv/spirv_vars.hpp>
#include <functional>
#include <memory>
#include <vector>
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

struct NativeCPUArgDesc {
  void *MPtr;

  NativeCPUArgDesc(void *Ptr) : MPtr(Ptr){};
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

struct __nativecpu_state {
  alignas(16) size_t MGlobal_id[3];
  alignas(16) size_t MGlobal_range[3];
  alignas(16) size_t MWorkGroup_size[3];
  alignas(16) size_t MWorkGroup_id[3];
  alignas(16) size_t MLocal_id[3];
  alignas(16) size_t MNumGroups[3];
  alignas(16) size_t MGlobalOffset[3];
  __nativecpu_state(size_t globalR0, size_t globalR1, size_t globalR2,
                    size_t localR0, size_t localR1, size_t localR2,
                    size_t globalO0, size_t globalO1, size_t globalO2)
      : MGlobal_range{globalR0, globalR1, globalR2},
        MWorkGroup_size{localR0, localR1, localR2},
        MNumGroups{globalR0 / localR0, globalR1 / localR1, globalR2 / localR2},
        MGlobalOffset{globalO0, globalO1, globalO2} {
    MGlobal_id[0] = 0;
    MGlobal_id[1] = 0;
    MGlobal_id[2] = 0;
    MWorkGroup_id[0] = 0;
    MWorkGroup_id[1] = 0;
    MWorkGroup_id[2] = 0;
    MLocal_id[0] = 0;
    MLocal_id[1] = 0;
    MLocal_id[2] = 0;
  }

  void update(size_t group0, size_t group1, size_t group2, size_t local0,
              size_t local1, size_t local2) {
    MWorkGroup_id[0] = group0;
    MWorkGroup_id[1] = group1;
    MWorkGroup_id[2] = group2;
    MLocal_id[0] = local0;
    MLocal_id[1] = local1;
    MLocal_id[2] = local2;
    MGlobal_id[0] = MWorkGroup_size[0] * MWorkGroup_id[0] + MLocal_id[0];
    MGlobal_id[1] = MWorkGroup_size[1] * MWorkGroup_id[1] + MLocal_id[1];
    MGlobal_id[2] = MWorkGroup_size[2] * MWorkGroup_id[2] + MLocal_id[2];
  }
};
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_HC_ATTRS                                                        \
  __attribute__((weak)) __attribute((alwaysinline))                            \
  [[intel::device_indirectly_callable]]

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_global_id(__attribute((address_space(0)))
                            __nativecpu_state *s) {
  return &(s->MGlobal_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_global_range(__attribute((address_space(0)))
                               __nativecpu_state *s) {
  return &(s->MGlobal_range[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_get_wg_size(__attribute((address_space(0)))
                              __nativecpu_state *s) {
  return &(s->MWorkGroup_size[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_get_wg_id(__attribute((address_space(0)))
                            __nativecpu_state *s) {
  return &(s->MWorkGroup_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_get_local_id(__attribute((address_space(0)))
                               __nativecpu_state *s) {
  return &(s->MLocal_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_get_num_groups(__attribute((address_space(0)))
                                 __nativecpu_state *s) {
  return &(s->MNumGroups[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
__dpcpp_nativecpu_get_global_offset(__attribute((address_space(0)))
                                    __nativecpu_state *s) {
  return &(s->MGlobalOffset[0]);
}
#undef __SYCL_HC_ATTRS
#endif
