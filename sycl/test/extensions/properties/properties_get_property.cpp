// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar));
  // Check that get_property of compile-time properties are the same as _v
  static_assert(P1::get_property<sycl::ext::oneapi::experimental::baz_key>() ==
                sycl::ext::oneapi::experimental::baz<1>);
  static_assert(P1::get_property<sycl::ext::oneapi::experimental::bar_key>() ==
                sycl::ext::oneapi::experimental::bar);

  // Check value on returned property
  static_assert(
      P1::get_property<sycl::ext::oneapi::experimental::baz_key>().value == 1);

  // Check runtime and compile-time properties on property-list object
  sycl::queue Q;
  sycl::ext::oneapi::experimental::foo FooProp{3};
  auto PropertyList = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar, FooProp);
  static_assert(decltype(PropertyList)::get_property<
                    sycl::ext::oneapi::experimental::baz_key>() ==
                sycl::ext::oneapi::experimental::baz<1>);
  static_assert(decltype(PropertyList)::get_property<
                    sycl::ext::oneapi::experimental::bar_key>() ==
                sycl::ext::oneapi::experimental::bar);
  assert(
      PropertyList.get_property<sycl::ext::oneapi::experimental::foo_key>() ==
      FooProp);
  assert(PropertyList.get_property<sycl::ext::oneapi::experimental::foo_key>()
             .value == FooProp.value);
  return 0;
}
