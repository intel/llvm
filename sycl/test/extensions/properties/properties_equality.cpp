// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar));
  using P2 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>));
  static_assert(std::is_same<P1, P2>::value);
}
