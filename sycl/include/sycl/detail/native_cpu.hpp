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
  nativecpu_state() {
    MGlobal_id[0] = 0;
    MGlobal_id[1] = 0;
    MGlobal_id[2] = 0;
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
#undef __SYCL_HC_ATTRS
#endif
