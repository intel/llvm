// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental;

int main() {
  // Check that oneapi::experimental::is_property_value is correctly specialized
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(cache_config{large_slm})>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(cache_config{large_data})>::value);

  // Check that property lists will accept the new properties
  sycl::ext::oneapi::experimental::properties Props1(
      cache_config{large_slm});
  using PS = decltype(Props1);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<PS>);
  static_assert(PS::has_property<cache_config>());
  assert(Props1.get_property<cache_config>() == large_slm);

  sycl::ext::oneapi::experimental::properties Props2(
      cache_config{large_data});
  using PS = decltype(Props2);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<PS>);
  static_assert(PS::has_property<cache_config>());
  assert(Props2.get_property<cache_config>() == large_data);

  // Check that duplicate cache_config can't be specified.
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Duplicate properties in property list.}}
  sycl::ext::oneapi::experimental::properties Props3(cache_config{large_data},
                                                     cache_config{large_slm});
}
