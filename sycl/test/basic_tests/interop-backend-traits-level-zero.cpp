// REQUIRES: level_zero
// RUN: %clangxx %fsycl-host-only -fsyntax-only %s

#include <sycl/sycl.hpp>
#include <ze_api.h>

constexpr auto Backend = sycl::backend::ext_oneapi_level_zero;

int main() {
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::platform>,
                     sycl::detail::interop<Backend, sycl::platform>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::platform>,
                     sycl::detail::interop<Backend, sycl::platform>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);

  return 0;
}
