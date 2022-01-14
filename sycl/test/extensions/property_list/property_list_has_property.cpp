// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Check has_property for compile-time properties
  using P1 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::baz::value_t<1>,
      sycl::ext::oneapi::experimental::bar::value_t>;
  static_assert(P1::has_property<sycl::ext::oneapi::experimental::bar>());
  static_assert(P1::has_property<sycl::ext::oneapi::experimental::baz>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::boo>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::foo>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::foz>());

  // Check has_property for runtime properties
  using P2 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::foo,
      sycl::ext::oneapi::experimental::foz>;
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::bar>());
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::baz>());
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::boo>());
  static_assert(P2::has_property<sycl::ext::oneapi::experimental::foo>());
  static_assert(P2::has_property<sycl::ext::oneapi::experimental::foz>());

  // Check has_property for a mix properties
  using P3 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::foo,
      sycl::ext::oneapi::experimental::baz::value_t<1>>;
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::bar>());
  static_assert(P3::has_property<sycl::ext::oneapi::experimental::baz>());
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::boo>());
  static_assert(P3::has_property<sycl::ext::oneapi::experimental::foo>());
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::foz>());
}
