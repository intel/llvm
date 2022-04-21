// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class NotAPropertyKey {};

int main() {
  // Check is_property_key for compile-time property keys.
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::bar_key>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::baz_key>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::boo_key>::value);

  // Check is_property_key for runtime property keys.
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::foo_key>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::foz_key>::value);

  // Check is_property_key for compile-time property values.
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<
                decltype(sycl::ext::oneapi::experimental::bar)>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<
                decltype(sycl::ext::oneapi::experimental::baz<42>)>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<
                decltype(sycl::ext::oneapi::experimental::boo<int>)>::value);
  static_assert(
      !sycl::ext::oneapi::experimental::is_property_key<
          decltype(sycl::ext::oneapi::experimental::boo<bool, float>)>::value);

  // Check is_property_key for runtime property values.
  // NOTE: For runtime properties the key is an alias of the value.
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::foo>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key<
                sycl::ext::oneapi::experimental::foz>::value);

  // Check is_property_key for non-property-key types.
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<int>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<
                NotAPropertyKey>::value);

  // Check is_property_key_of for compile-time property keys.
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::bar_key, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::baz_key, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::boo_key, sycl::queue>::value);

  // Check is_property_key_of for runtime property keys.
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::foo_key, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::foz_key, sycl::queue>::value);

  // Check is_property_key_of for compile-time property values.
  static_assert(!sycl::ext::oneapi::experimental::is_property_key<
                decltype(sycl::ext::oneapi::experimental::bar)>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key_of<
                decltype(sycl::ext::oneapi::experimental::baz<42>),
                sycl::queue>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key_of<
                decltype(sycl::ext::oneapi::experimental::boo<int>),
                sycl::queue>::value);
  static_assert(!sycl::ext::oneapi::experimental::is_property_key_of<
                decltype(sycl::ext::oneapi::experimental::boo<bool, float>),
                sycl::queue>::value);

  // Check is_property_key_of for runtime property keys.
  // NOTE: For runtime properties the key is an alias of the value.
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::foo, sycl::queue>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_key_of<
                sycl::ext::oneapi::experimental::foz, sycl::queue>::value);

  // Check is_property_key_of for non-property-key types.
  static_assert(
      !sycl::ext::oneapi::experimental::is_property_key_of<int,
                                                           sycl::queue>::value);
  static_assert(
      !sycl::ext::oneapi::experimental::is_property_key_of<NotAPropertyKey,
                                                           sycl::queue>::value);

  return 0;
}
