// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  using P1 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::ext::oneapi::bar::value_t>;
  // Check that get_property of compile-time properties are the same as _v
  static_assert(P1::get_property<sycl::ext::oneapi::baz>() ==
                sycl::ext::oneapi::baz_v<1>);
  static_assert(P1::get_property<sycl::ext::oneapi::bar>() ==
                sycl::ext::oneapi::bar_v);

  // Check value on returned property
  static_assert(P1::get_property<sycl::ext::oneapi::baz>().value == 1);

  // Check runtime and compile-time properties on property-list object
  sycl::queue Q;
  auto PropertyList = sycl::ext::oneapi::property_list(
      sycl::ext::oneapi::baz_v<1>, sycl::ext::oneapi::bar_v,
      sycl::property::no_init{},
      sycl::property::buffer::context_bound{Q.get_context()});
  static_assert(
      decltype(PropertyList)::get_property<sycl::ext::oneapi::baz>() ==
      sycl::ext::oneapi::baz_v<1>);
  static_assert(
      decltype(PropertyList)::get_property<sycl::ext::oneapi::bar>() ==
      sycl::ext::oneapi::bar_v);
  assert(PropertyList.get_property<sycl::property::buffer::context_bound>()
             .get_context() == Q.get_context());
}
