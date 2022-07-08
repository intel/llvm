// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // expected-error-re@sycl/ext/oneapi/properties/property_utils.hpp:* {{static_assert failed due to requirement {{.+}} "Unrecognized property in property list."}}
  // expected-error@+1 {{no viable constructor or deduction guide for deduction of template arguments of 'properties'}}
  auto InvalidPropertyList1 = sycl::ext::oneapi::experimental::properties(1);
  // expected-error-re@sycl/ext/oneapi/properties/property_utils.hpp:* {{static_assert failed due to requirement {{.+}} "Unrecognized property in property list."}}
  // expected-error@+1 {{no viable constructor or deduction guide for deduction of template arguments of 'properties'}}
  auto InvalidPropertyList2 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{1}, true);
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement {{.+}} "Duplicate properties in property list."}}
  auto InvalidPropertyList3 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::foo{1});
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement {{.+}} "Duplicate properties in property list."}}
  auto InvalidPropertyList4 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::bar);
  return 0;
}
