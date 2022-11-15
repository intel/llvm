//==--- cache_read.hpp - SYCL_ONEAPI_CUDA_CACHE_READ  ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define SYCL_EXT_ONEAPI_CUDA_CACHE_READ 1

#include <sycl/ext/oneapi/experimental/bfloat16.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace cuda {

using ldg_scalar_types =
    sycl::detail::type_list<sycl::detail::gtl::scalar_signed_basic_list,
                            sycl::detail::gtl::scalar_unsigned_basic_list, sycl::ext::oneapi::experimental::bfloat16>;

template <typename T>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::is_contained<T, ldg_scalar_types>::value, T>
    cache_read(const T *ptr) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
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
  } else if constexpr (std::is_same_v<
                           T, sycl::ext::oneapi::experimental::bfloat16>) {
    return bfloat16::from_bits(
        __nvvm_ldg_us((reinterpret_cast<const unsigned short *>(ptr))));
  } else if constexpr (std::is_same_v<T, float>) {
    return __nvvm_ldg_f(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return __nvvm_ldg_d(ptr);
  }
#else
  throw runtime_error("ldg is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

} // namespace cuda
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
