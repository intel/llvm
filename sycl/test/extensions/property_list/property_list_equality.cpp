// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  using P1 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::ext::oneapi::bar::value_t>;
  using P2 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::bar::value_t,
                                         sycl::ext::oneapi::baz::value_t<1>>;
  static_assert(std::is_same<P1, P2>::value);
}
