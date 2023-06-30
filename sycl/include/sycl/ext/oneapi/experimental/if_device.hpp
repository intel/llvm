#pragma once

#include <sycl/detail/common.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {
namespace detail {
// Helper object used to implement "otherwise".  The "MakeCall" template
// parameter tells whether the previous call to "if_device" or "if_host" called
// its "fn".  When "MakeCall" is true, the previous call to "fn" did not
// happen, so the "otherwise" should call "fn".
template <bool MakeCall> class if_device_or_host_helper {
public:
  template <typename T> void otherwise(T fn) {
    if constexpr (MakeCall) {
      fn();
    }
  }
};
} // namespace detail

template <typename T> static auto if_device(T fn) {
#ifdef __SYCL_DEVICE_ONLY__
  fn();
  return detail::if_device_or_host_helper<false>{};
#else
  return detail::if_device_or_host_helper<true>{};
#endif
}

template <typename T> static auto if_host(T fn) {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::if_device_or_host_helper<true>{};
#else
  fn();
  return detail::if_device_or_host_helper<false>{};
#endif
}
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
