// REQUIRES: opencl
// RUN: %clangxx -fsycl -fsyntax-only %s

#include <CL/cl.h>
#include <sycl/sycl.hpp>

constexpr auto Backend = sycl::backend::opencl;

int main() {
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::device>,
                     sycl::detail::interop<Backend, sycl::device>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::context>,
                     sycl::detail::interop<Backend, sycl::context>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::queue>,
                     sycl::detail::interop<Backend, sycl::queue>::type>);
  static_assert(
      std::is_same_v<sycl::backend_traits<Backend>::input_type<sycl::event>,
                     sycl::detail::interop<Backend, sycl::event>::type>);
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
