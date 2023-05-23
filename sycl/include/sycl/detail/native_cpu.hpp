#pragma once
#include "cg_types.hpp"
#include <CL/__spirv/spirv_vars.hpp>
#include <functional>
#include <memory>
#include <vector>
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

struct __SYCL_EXPORT NativeCPUArgDesc {
  void *MPtr;

  void *getPtr() const { return MPtr; }
  NativeCPUArgDesc(void *Ptr) : MPtr(Ptr){};
};



} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

struct nativecpu_state {
  size_t MGlobal_id[3];
  size_t MGlobal_range[3];
  size_t MWorkGroup_size[3];
  size_t MWorkGroup_id[3];
  size_t MLocal_id[3];
  size_t MNumGroups[3];
  size_t MGlobalOffset[3];
  nativecpu_state(size_t globalR0, size_t globalR1, size_t globalR2,
                  size_t localR0, size_t localR1, size_t localR2,
                  size_t globalO0, size_t globalO1, size_t globalO2)
      : MGlobal_range{globalR0, globalR1, globalR2}, MWorkGroup_size{localR0,
                                                                     localR1,
                                                                     localR2},
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
_Z13get_global_idmP15nativecpu_state(__attribute((address_space(0)))
                                     nativecpu_state *s) {
  return &(s->MGlobal_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_global_rangemP15nativecpu_state(__attribute((address_space(0)))
                                        nativecpu_state *s) {
  return &(s->MGlobal_range[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_wg_sizemP15nativecpu_state(__attribute((address_space(0)))
                                   nativecpu_state *s) {
  return &(s->MWorkGroup_size[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_wgid_mP15nativecpu_state(__attribute((address_space(0)))
                                 nativecpu_state *s) {
  return &(s->MWorkGroup_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_local_id_mP15nativecpu_state(__attribute((address_space(0)))
                                     nativecpu_state *s) {
  return &(s->MLocal_id[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_num_groupsmP15nativecpu_state(__attribute((address_space(0)))
                                      nativecpu_state *s) {
  return &(s->MNumGroups[0]);
}

extern "C" __SYCL_HC_ATTRS __attribute((address_space(0))) size_t *
_Z13get_global_offsetmP15nativecpu_state(__attribute((address_space(0)))
                                         nativecpu_state *s) {
  return &(s->MGlobalOffset[0]);
}
#undef __SYCL_HC_ATTRS
#endif
