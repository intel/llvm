// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi;

int main() {
  {
    // Single RT property
    accessor_property_list PL{sycl::no_init};
    static_assert(!PL.has_property<property::no_offset>(), "Property is found");
    assert(PL.has_property<sycl::property::no_init>() && "Property not found");
  }

  {
    // Single RT property
    accessor_property_list PL{sycl::noinit};
    static_assert(!PL.has_property<property::no_offset>(), "Property is found");
    assert(PL.has_property<sycl::property::noinit>() && "Property not found");
  }

  {
    // Compile time and runtime properties
    accessor_property_list PL{sycl::no_init, no_alias};
    assert(PL.has_property<property::no_alias>() && "Property not found");
    assert(PL.has_property<sycl::property::no_init>() && "Property not found");
  }

  {
    // Compile time and runtime properties
    accessor_property_list PL{sycl::noinit, no_alias};
    assert(PL.has_property<property::no_alias>() && "Property not found");
    assert(PL.has_property<sycl::property::noinit>() && "Property not found");
  }
  return 0;
}
