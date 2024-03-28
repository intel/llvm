#include <sycl/id.hpp>
#include <sycl/exception.hpp> // for make_error_code, errc, exce...
#include <sycl/detail/common.hpp>             // for InitializedVal

namespace sycl {
inline namespace _V1 {
  template <int Dimensions>
  __SYCL_DEPRECATED("range() conversion is deprecated")
  id<Dimensions>::operator range<Dimensions>() const {
    range<Dimensions> result(
        detail::InitializedVal<Dimensions, range>::template get<0>());
    for (int i = 0; i < Dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }


template <int Dims>
__SYCL_DEPRECATED("use sycl::ext::oneapi::experimental::this_id() instead")
id<Dims> this_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::Builder::getElement(detail::declptr<id<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

namespace ext::oneapi::experimental {
template <int Dims> id<Dims> this_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(sycl::detail::declptr<id<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
