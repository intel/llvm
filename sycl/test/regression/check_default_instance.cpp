// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

int main() {
  sycl::ext::oneapi::accessor_property_list PL{
      sycl::ext::intel::buffer_location<1>};
  static_assert(PL.has_property<sycl::ext::intel::property::buffer_location>(),
                "Property not found");
  return 0;
}
