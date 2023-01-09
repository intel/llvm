//==--- builtins.hpp - SYCL_ONEAPI_CUDA experimental builtins  -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define SYCL_EXT_ONEAPI_CUDA_TEX_CACHE_READ 1

#include <sycl/types.hpp>

#if defined(_WIN32) || defined(_WIN64)
#define ATTRIBUTE_EXT_VEC_TYPE(N) __declspec(ext_vector_type(N))
#else
#define ATTRIBUTE_EXT_VEC_TYPE(N) __attribute__((ext_vector_type(N)))
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace cuda {

namespace detail {
using ldg_types = sycl::detail::type_list<float, double, sycl::float2,
                                          sycl::float4, sycl::double2>;
} // namespace detail

template <typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    sycl::detail::is_contained<
        T, sycl::ext::oneapi::experimental::cuda::detail::ldg_types>::value,
    T>
ldg(const T *ptr) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  if constexpr (std::is_same_v<T, float>) {
    return __nvvm_ldg_f(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return __nvvm_ldg_d(ptr);
  } else if constexpr (std::is_same_v<T, sycl::float2>) {
    // We can assume that ptr is aligned at least to float2's alignment, but the
    // load will assume that ptr is aligned to float2's alignment.  This is only
    // safe if alignof(f2) <= alignof(float2).
    typedef float f2 ATTRIBUTE_EXT_VEC_TYPE(2);
    f2 rv = __nvvm_ldg_f2(reinterpret_cast<const f2 *>(ptr));
    sycl::float2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::float4>) {
    typedef float f4 ATTRIBUTE_EXT_VEC_TYPE(4);
    f4 rv = __nvvm_ldg_f4(reinterpret_cast<const f4 *>(ptr));
    sycl::float4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::double2>) {
    typedef double d2 ATTRIBUTE_EXT_VEC_TYPE(2);
    d2 rv = __nvvm_ldg_d2(reinterpret_cast<const d2 *>(ptr));
    sycl::double2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  }
#else
  return *ptr;
#endif
#else
  throw runtime_error("ldg is not supported on host.", PI_ERROR_INVALID_DEVICE);
#endif
}

#undef ATTRIBUTE_EXT_VEC_TYPE(N)

} // namespace cuda
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
