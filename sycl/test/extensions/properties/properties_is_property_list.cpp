// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Check is_property_list for empty property list
  using EmptyP = decltype(sycl::ext::oneapi::experimental::properties());
  static_assert(
      sycl::ext::oneapi::experimental::is_property_list<EmptyP>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<EmptyP>);

  // Check is_property_list for compile-time properties
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar));
  static_assert(sycl::ext::oneapi::experimental::is_property_list<P1>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<P1>);

  // Check is_property_list for runtime properties
  using P2 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{42},
      sycl::ext::oneapi::experimental::foz{3.14159265, false}));
  static_assert(sycl::ext::oneapi::experimental::is_property_list<P2>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<P2>);

  // Check is_property_list for a mix properties
  using P3 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::foz{1.2, true}));
  static_assert(sycl::ext::oneapi::experimental::is_property_list<P3>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<P3>);

  // Check is_property_list for compile-time properties on object
  auto PropertyList1 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::bar);
  static_assert(sycl::ext::oneapi::experimental::is_property_list<
                decltype(PropertyList1)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<
                decltype(PropertyList1)>);

  // Check is_property_list for runtime properties on object
  sycl::queue Q;
  auto PropertyList2 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{1},
      sycl::ext::oneapi::experimental::foz{.123f, false});
  static_assert(sycl::ext::oneapi::experimental::is_property_list<
                decltype(PropertyList2)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<
                decltype(PropertyList2)>);

  // Check is_property_list for a mix properties on object
  auto PropertyList3 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{42},
      sycl::ext::oneapi::experimental::baz<1>);
  static_assert(sycl::ext::oneapi::experimental::is_property_list<
                decltype(PropertyList3)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<
                decltype(PropertyList3)>);

  // Check is_property_list on types that are not valid
  // sycl::ext::oneapi::experimental::property_list.
  static_assert(!sycl::ext::oneapi::experimental::is_property_list<int>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_list_v<int>);
  static_assert(!sycl::ext::oneapi::experimental::is_property_list<
                sycl::property_list>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_list_v<
                sycl::property_list>);

  return 0;
}
