// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Check is_property_list for empty property list
  using EmptyP = sycl::ext::oneapi::property_list_t<>;
  static_assert(sycl::ext::oneapi::is_property_list<EmptyP>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<EmptyP>);

  // Check is_property_list for compile-time properties
  using P1 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::ext::oneapi::bar::value_t>;
  static_assert(sycl::ext::oneapi::is_property_list<P1>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<P1>);

  // Check is_property_list for runtime properties
  using P2 =
      sycl::ext::oneapi::property_list_t<sycl::property::buffer::use_host_ptr,
                                         sycl::property::image::context_bound>;
  static_assert(sycl::ext::oneapi::is_property_list<P2>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<P2>);

  // Check is_property_list for a mix properties
  using P3 =
      sycl::ext::oneapi::property_list_t<sycl::property::buffer::use_host_ptr,
                                         sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::property::image::context_bound>;
  static_assert(sycl::ext::oneapi::is_property_list<P3>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<P3>);

  // Check is_property_list for compile-time properties on object
  auto PropertyList1 = sycl::ext::oneapi::property_list(
      sycl::ext::oneapi::baz_v<1>, sycl::ext::oneapi::bar_v);
  static_assert(
      sycl::ext::oneapi::is_property_list<decltype(PropertyList1)>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<decltype(PropertyList1)>);

  // Check is_property_list for runtime properties on object
  sycl::queue Q;
  auto PropertyList2 = sycl::ext::oneapi::property_list(
      sycl::property::buffer::use_host_ptr{},
      sycl::property::image::context_bound{Q.get_context()});
  static_assert(
      sycl::ext::oneapi::is_property_list<decltype(PropertyList2)>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<decltype(PropertyList2)>);

  // Check is_property_list for a mix properties on object
  auto PropertyList3 = sycl::ext::oneapi::property_list(
      sycl::property::buffer::use_host_ptr{}, sycl::ext::oneapi::baz_v<1>,
      sycl::property::image::context_bound{Q.get_context()});
  static_assert(
      sycl::ext::oneapi::is_property_list<decltype(PropertyList3)>::value);
  static_assert(sycl::ext::oneapi::is_property_list_v<decltype(PropertyList3)>);

  // Check is_property_list on types that are not valid
  // sycl::ext::oneapi::property_list.
  static_assert(!sycl::ext::oneapi::is_property_list<int>::value);
  static_assert(!sycl::ext::oneapi::is_property_list_v<int>);
  static_assert(
      !sycl::ext::oneapi::is_property_list<sycl::property_list>::value);
  static_assert(!sycl::ext::oneapi::is_property_list_v<sycl::property_list>);
}
