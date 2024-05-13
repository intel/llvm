// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental;

int main() {
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
  // TODO: Do we want the base class error?
  // Pro: tells user which runtime property is duplicated and allows property
  // list to be zero-sized if it doesn't contain any runtime properties. Cons:
  // error might be confusing If we don't want it the storage needs to be moved
  // from base class to member.
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{base class 'sycl::ext::intel::experimental::cache_config' specified more than once as a direct base class}}
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Duplicate properties in property list.}}
  sycl::ext::oneapi::experimental::properties Props3(cache_config{large_data},
                                                     cache_config{large_slm});
}
