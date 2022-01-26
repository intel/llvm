// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

int main() {
  sycl::ext::oneapi::accessor_property_list PL{sycl::ext::oneapi::no_alias};
  static_assert(
      decltype(PL)::has_property<sycl::ext::oneapi::property::no_alias>(),
      "Property is not found");
  static_assert(
      decltype(PL)::get_property<sycl::ext::oneapi::property::no_alias>() ==
          sycl::ext::oneapi::no_alias,
      "Properties are not equal");
  return 0;
}
