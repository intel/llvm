// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  auto EmptyPropertyList = sycl::ext::oneapi::experimental::properties();
  // expected-error@+1 {{no matching function for call to 'get_property'}}
  constexpr auto boo_val1 = decltype(EmptyPropertyList)::get_property<
      sycl::ext::oneapi::experimental::boo_key>();
  // expected-error@+3 {{no matching member function for call to 'get_property'}}
  sycl::ext::oneapi::experimental::foo foo_val1 =
      EmptyPropertyList
          .get_property<sycl::ext::oneapi::experimental::foo_key>();

  sycl::queue Q;
  auto PopulatedPropertyList = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::foz{.0f, true},
      sycl::ext::oneapi::experimental::bar);
  // expected-error@+1 {{no matching function for call to 'get_property'}}
  constexpr auto boo_val2 = decltype(PopulatedPropertyList)::get_property<
      sycl::ext::oneapi::experimental::boo_key>();
  // expected-error@+3 {{no matching member function for call to 'get_property'}}
  sycl::ext::oneapi::experimental::foo foo_val2 =
      PopulatedPropertyList
          .get_property<sycl::ext::oneapi::experimental::foo_key>();
  return 0;
}
