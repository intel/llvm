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
using ldg_vector_types = sycl::detail::type_list<
    sycl::char2, sycl::char4, sycl::short2, sycl::short4, sycl::int2,
    sycl::int4, sycl::longlong2, sycl::uchar2, sycl::uchar4, sycl::ushort2,
    sycl::ushort4, sycl::uint2, sycl::uint4, sycl::ulonglong2, sycl::float2,
    sycl::float4, sycl::double2>;

using ldg_types =
    sycl::detail::type_list<ldg_vector_types,
                            sycl::detail::gtl::scalar_signed_basic_list,
                            sycl::detail::gtl::scalar_unsigned_basic_list>;
} // namespace detail

template <typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    sycl::detail::is_contained<
        T, sycl::ext::oneapi::experimental::cuda::detail::ldg_types>::value,
    T>
ldg(const T *ptr) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  if constexpr (std::is_same_v<T, char>) {
    return __nvvm_ldg_c(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return __nvvm_ldg_s(ptr);
  } else if constexpr (std::is_same_v<T, int>) {
    return __nvvm_ldg_i(ptr);
  } else if constexpr (std::is_same_v<T, long>) {
    return __nvvm_ldg_l(ptr);
  } else if constexpr (std::is_same_v<T, long long>) {
    return __nvvm_ldg_ll(ptr);
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return __nvvm_ldg_uc(ptr);
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    return __nvvm_ldg_us(ptr);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __nvvm_ldg_ui(ptr);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __nvvm_ldg_ul(ptr);
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return __nvvm_ldg_ull(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return __nvvm_ldg_f(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return __nvvm_ldg_d(ptr);
  } else if constexpr (std::is_same_v<T, sycl::char2>) {
    // We can assume that ptr is aligned at least to char2's alignment, but the
    // load will assume that ptr is aligned to char2's alignment.  This is only
    // safe if alignof(f2) <= alignof(char2).
    typedef char c2 ATTRIBUTE_EXT_VEC_TYPE(2);
    c2 rv = __nvvm_ldg_c2(reinterpret_cast<const c2 *>(ptr));
    sycl::char2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::char4>) {
    typedef char c4 ATTRIBUTE_EXT_VEC_TYPE(4);
    c4 rv = __nvvm_ldg_c4(reinterpret_cast<const c4 *>(ptr));
    sycl::char4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::short2>) {
    typedef short s2 ATTRIBUTE_EXT_VEC_TYPE(2);
    s2 rv = __nvvm_ldg_s2(reinterpret_cast<const s2 *>(ptr));
    sycl::short2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::short4>) {
    typedef short s4 ATTRIBUTE_EXT_VEC_TYPE(4);
    s4 rv = __nvvm_ldg_s4(reinterpret_cast<const s4 *>(ptr));
    sycl::short4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::int2>) {
    typedef int i2 ATTRIBUTE_EXT_VEC_TYPE(2);
    i2 rv = __nvvm_ldg_i2(reinterpret_cast<const i2 *>(ptr));
    sycl::int2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::int4>) {
    typedef int i4 ATTRIBUTE_EXT_VEC_TYPE(4);
    i4 rv = __nvvm_ldg_i4(reinterpret_cast<const i4 *>(ptr));
    sycl::int4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::longlong2>) {
    typedef long long ll2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ll2 rv = __nvvm_ldg_ll2(reinterpret_cast<const ll2 *>(ptr));
    sycl::longlong2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::uchar2>) {
    typedef unsigned char uc2 ATTRIBUTE_EXT_VEC_TYPE(2);
    uc2 rv = __nvvm_ldg_uc2(reinterpret_cast<const uc2 *>(ptr));
    sycl::uchar2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::uchar4>) {
    typedef unsigned char uc4 ATTRIBUTE_EXT_VEC_TYPE(4);
    uc4 rv = __nvvm_ldg_uc4(reinterpret_cast<const uc4 *>(ptr));
    sycl::uchar4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::ushort2>) {
    typedef unsigned short us2 ATTRIBUTE_EXT_VEC_TYPE(2);
    us2 rv = __nvvm_ldg_us2(reinterpret_cast<const us2 *>(ptr));
    sycl::ushort2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::ushort4>) {
    typedef unsigned short us4 ATTRIBUTE_EXT_VEC_TYPE(4);
    us4 rv = __nvvm_ldg_us4(reinterpret_cast<const us4 *>(ptr));
    sycl::ushort4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::uint2>) {
    typedef unsigned int ui2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ui2 rv = __nvvm_ldg_ui2(reinterpret_cast<const ui2 *>(ptr));
    sycl::uint2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::uint4>) {
    typedef unsigned int ui4 ATTRIBUTE_EXT_VEC_TYPE(4);
    ui4 rv = __nvvm_ldg_ui4(reinterpret_cast<const ui4 *>(ptr));
    sycl::uint4 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::ulonglong2>) {
    typedef unsigned long long ull2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ull2 rv = __nvvm_ldg_ull2(reinterpret_cast<const ull2 *>(ptr));
    sycl::ulonglong2 ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::float2>) {
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

#undef ATTRIBUTE_EXT_VEC_TYPE

} // namespace cuda
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
