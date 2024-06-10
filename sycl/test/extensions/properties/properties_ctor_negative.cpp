// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // expected-error-re@sycl/ext/oneapi/properties/property_utils.hpp:* {{static assertion failed due to requirement {{.+}}: Unrecognized property in property list.}}
  // expected-error@+1 {{no viable constructor or deduction guide for deduction of template arguments of 'properties'}}
  auto InvalidPropertyList1 = sycl::ext::oneapi::experimental::properties(1);
  // expected-error-re@sycl/ext/oneapi/properties/property_utils.hpp:* {{static assertion failed due to requirement {{.+}}: Unrecognized property in property list.}}
  // expected-error@+1 {{no viable constructor or deduction guide for deduction of template arguments of 'properties'}}
  auto InvalidPropertyList2 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{1}, true);
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Duplicate properties in property list.}}
  auto InvalidPropertyList3 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::foo{1});
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Duplicate properties in property list.}}
  auto InvalidPropertyList4 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::bar);
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  auto InvalidPropertyList5 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::fir(3.14, false));
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  auto InvalidPropertyList6 = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::fir(3.14, false));
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: One or more property argument is not a property in the property list.}}
  sycl::ext::oneapi::experimental::empty_properties_t InvalidPropertyList7{
      sycl::ext::oneapi::experimental::foo{0}};
  /*
  // TODO: Uncomment these lines once https://github.com/intel/llvm/issues/13677
  // has been fixed.
  decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::foo{0}}) InvalidPropertyList8{
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::foo{1}};
  decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::boo<int>}) InvalidPropertyList9{
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::boo<int>};
  decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::foo{0}}) InvalidPropertyList10{
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::foo{1}};
  decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::boo<int>}) InvalidPropertyList11{
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::boo<int>};
  */

  // TODO: For the following cases, the second error could be removed by moving
  //       the static assert out of the Extract function. However, this
  //       currently causes the same Clang crash as above.
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Each property in the property list must either be given an argument in the constructor or be default-constructible.}}
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{no matching constructor for initialization of 'sycl::ext::oneapi::experimental::fir'}}
  decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::fir(3.14, false))) InvalidPropertyList12{
      sycl::ext::oneapi::experimental::bar};
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Each property in the property list must either be given an argument in the constructor or be default-constructible.}}
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{no matching constructor for initialization of 'sycl::ext::oneapi::experimental::fir'}}
  decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::fir(3.14,
                                           false))) InvalidPropertyList13{};
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Each property in the property list must either be given an argument in the constructor or be default-constructible.}}
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{no matching constructor for initialization of 'sycl::ext::oneapi::experimental::fir'}}
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: One or more property argument is not a property in the property list.}}
  decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::fir(3.14, false))) InvalidPropertyList14{
      sycl::ext::oneapi::experimental::boo<int>,
      sycl::ext::oneapi::experimental::bar};
  // expected-error@+2 {{no matching constructor for initialization of }}
  decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foo{1})) InvalidPropertyList15{1};
  // expected-error@+3 {{no matching constructor for initialization of }}
  decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foo{1})) InvalidPropertyList16{
      sycl::ext::oneapi::experimental::bar, 1};
  return 0;
}
