// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  using P1 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::baz::value_t<1>,
      sycl::ext::oneapi::experimental::bar::value_t>;
  // Check that get_property of compile-time properties are the same as _v
  static_assert(P1::get_property<sycl::ext::oneapi::experimental::baz>() ==
                sycl::ext::oneapi::experimental::baz_v<1>);
  static_assert(P1::get_property<sycl::ext::oneapi::experimental::bar>() ==
                sycl::ext::oneapi::experimental::bar_v);

  // Check value on returned property
  static_assert(
      P1::get_property<sycl::ext::oneapi::experimental::baz>().value == 1);

  // Check runtime and compile-time properties on property-list object
  sycl::queue Q;
  sycl::ext::oneapi::experimental::foo FooProp{3};
  auto PropertyList = sycl::ext::oneapi::experimental::property_list(
      sycl::ext::oneapi::experimental::baz_v<1>,
      sycl::ext::oneapi::experimental::bar_v, FooProp);
  static_assert(decltype(PropertyList)::get_property<
                    sycl::ext::oneapi::experimental::baz>() ==
                sycl::ext::oneapi::experimental::baz_v<1>);
  static_assert(decltype(PropertyList)::get_property<
                    sycl::ext::oneapi::experimental::bar>() ==
                sycl::ext::oneapi::experimental::bar_v);
  assert(PropertyList.get_property<sycl::ext::oneapi::experimental::foo>() ==
         FooProp);
  assert(
      PropertyList.get_property<sycl::ext::oneapi::experimental::foo>().value ==
      FooProp.value);
}
