// REQUIRES: hip
// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

constexpr auto Backend = sycl::backend::ext_oneapi_hip;

int main() {
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::return_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);

  return 0;
}
