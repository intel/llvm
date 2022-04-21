// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class NotAPropertyKey {};

int main() {
  // Check is_property_value for compile-time property values.
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::oneapi::experimental::bar)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::oneapi::experimental::baz<42>)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::oneapi::experimental::boo<int>)>::value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::oneapi::experimental::boo<bool, float>)>::value);

  // Check is_property_value for runtime property values.
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::foo>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::foz>::value);

  // Check is_property_value for compile-time property keys.
  static_assert(!sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::bar_key>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::baz_key>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::boo_key>::value);

  // Check is_property_value for runtime property keys.
  // NOTE: For runtime properties the key is an alias of the value.
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::foo_key>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                sycl::ext::oneapi::experimental::foz_key>::value);

  // Check is_property_value for non-property-key types.
  static_assert(
      !sycl::ext::oneapi::experimental::is_property_value<int>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value<
                NotAPropertyKey>::value);

  // Check is_property_value_of for compile-time property values.
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::oneapi::experimental::bar)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                decltype(sycl::ext::oneapi::experimental::baz<42>),
                sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                decltype(sycl::ext::oneapi::experimental::boo<int>),
                sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                decltype(sycl::ext::oneapi::experimental::boo<bool, float>),
                sycl::queue>::value);

  // Check is_property_value_of for runtime property keys.
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::foo, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::foz, sycl::queue>::value);

  // Check is_property_value_of for compile-time property keys.
  static_assert(!sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::bar_key, sycl::queue>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::baz_key, sycl::queue>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::boo_key, sycl::queue>::value);

  // Check is_property_value_of for runtime property keys.
  // NOTE: For runtime properties the key is an alias of the value.
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::foo_key, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value_of<
                sycl::ext::oneapi::experimental::foz_key, sycl::queue>::value);

  // Check is_property_value_of for non-property-key types.
  static_assert(!sycl::ext::oneapi::experimental::is_property_value_of<
                int, sycl::queue>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_value_of<
                NotAPropertyKey, sycl::queue>::value);

  return 0;
}
