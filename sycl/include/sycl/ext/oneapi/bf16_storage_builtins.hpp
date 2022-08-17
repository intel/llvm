#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/builtins.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/detail/generic_type_lists.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

namespace detail {

template <typename T> struct is_bf16_storage_type {
  static constexpr int value = false;
};

template <> struct is_bf16_storage_type<uint16_t> {
  static constexpr int value = true;
};

template <> struct is_bf16_storage_type<uint32_t> {
  static constexpr int value = true;
};

template <int N> struct is_bf16_storage_type<vec<uint16_t, N>> {
  static constexpr int value = true;
};

template <int N> struct is_bf16_storage_type<vec<uint32_t, N>> {
  static constexpr int value = true;
};

} // namespace detail

template <typename T>
std::enable_if_t<detail::is_bf16_storage_type<T>::value, T> fabs(T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fabs(x);
#else
  (void)x;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
template <typename T>
std::enable_if_t<detail::is_bf16_storage_type<T>::value, T> fmin(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fmin(x, y);
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
template <typename T>
std::enable_if_t<detail::is_bf16_storage_type<T>::value, T> fmax(T x, T y) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fmax(x, y);
#else
  (void)x;
  (void)y;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
template <typename T>
std::enable_if_t<detail::is_bf16_storage_type<T>::value, T> fma(T x, T y, T z) {
#ifdef __SYCL_DEVICE_ONLY__
  return __clc_fma(x, y, z);
#else
  (void)x;
  (void)y;
  (void)z;
  throw runtime_error("bf16 is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
