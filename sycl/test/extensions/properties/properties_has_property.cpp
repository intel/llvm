// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Check has_property for compile-time properties
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar));
  static_assert(P1::has_property<sycl::ext::oneapi::experimental::bar_key>());
  static_assert(P1::has_property<sycl::ext::oneapi::experimental::baz_key>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::boo_key>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::foo_key>());
  static_assert(!P1::has_property<sycl::ext::oneapi::experimental::foz_key>());

  // Check has_property for runtime properties
  using P2 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{2},
      sycl::ext::oneapi::experimental::foz{1.23, true}));
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::bar_key>());
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::baz_key>());
  static_assert(!P2::has_property<sycl::ext::oneapi::experimental::boo_key>());
  static_assert(P2::has_property<sycl::ext::oneapi::experimental::foo_key>());
  static_assert(P2::has_property<sycl::ext::oneapi::experimental::foz_key>());

  // Check has_property for a mix properties
  using P3 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{1000},
      sycl::ext::oneapi::experimental::baz<1>));
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::bar_key>());
  static_assert(P3::has_property<sycl::ext::oneapi::experimental::baz_key>());
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::boo_key>());
  static_assert(P3::has_property<sycl::ext::oneapi::experimental::foo_key>());
  static_assert(!P3::has_property<sycl::ext::oneapi::experimental::foz_key>());

  return 0;
}
