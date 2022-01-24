// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  auto EmptyPropertyList = sycl::ext::oneapi::experimental::properties();
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement 'has_property<sycl::ext::oneapi::experimental::boo_key>()' "Property list does not contain the requested property."}}
  // expected-error@+1 {{variable has incomplete type 'const void'}}
  constexpr auto boo_val1 = decltype(EmptyPropertyList)::get_property<
      sycl::ext::oneapi::experimental::boo_key>();
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement 'has_property<sycl::ext::oneapi::experimental::foo>()' "Property list does not contain the requested property."}}
  // expected-error@+1 {{no viable conversion from 'typename std::enable_if_t<detail::IsRuntimeProperty<foo>::value && !has_property<foo>(), void>' (aka 'void') to 'sycl::ext::oneapi::experimental::foo'}}
  sycl::ext::oneapi::experimental::foo foo_val1 =
      EmptyPropertyList
          .get_property<sycl::ext::oneapi::experimental::foo_key>();

  sycl::queue Q;
  auto PopulatedPropertyList = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foz{.0f, true},
      sycl::ext::oneapi::experimental::bar);
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement 'has_property<sycl::ext::oneapi::experimental::boo_key>()' "Property list does not contain the requested property."}}
  // expected-error@+1 {{variable has incomplete type 'const void'}}
  constexpr auto boo_val2 = decltype(PopulatedPropertyList)::get_property<
      sycl::ext::oneapi::experimental::boo_key>();
  // expected-error@sycl/ext/oneapi/properties/properties.hpp:* {{static_assert failed due to requirement 'has_property<sycl::ext::oneapi::experimental::foo>()' "Property list does not contain the requested property."}}
  // expected-error@+1 {{no viable conversion from 'typename std::enable_if_t<detail::IsRuntimeProperty<foo>::value && !has_property<foo>(), void>' (aka 'void') to 'sycl::ext::oneapi::experimental::foo'}}
  sycl::ext::oneapi::experimental::foo foo_val2 =
      PopulatedPropertyList
          .get_property<sycl::ext::oneapi::experimental::foo_key>();
  return 0;
}
