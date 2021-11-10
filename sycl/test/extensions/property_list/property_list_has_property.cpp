// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Check has_property for compile-time properties
  using P1 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::ext::oneapi::bar::value_t>;
  static_assert(P1::has_property<sycl::ext::oneapi::bar>());
  static_assert(P1::has_property<sycl::ext::oneapi::baz>());
  static_assert(!P1::has_property<sycl::ext::oneapi::boo>());
  static_assert(!P1::has_property<sycl::property::buffer::use_host_ptr>());
  static_assert(!P1::has_property<sycl::property::image::context_bound>());

  // Check has_property for runtime properties
  using P2 =
      sycl::ext::oneapi::property_list_t<sycl::property::buffer::use_host_ptr,
                                         sycl::property::image::context_bound>;
  static_assert(!P2::has_property<sycl::ext::oneapi::bar>());
  static_assert(!P2::has_property<sycl::ext::oneapi::baz>());
  static_assert(!P2::has_property<sycl::ext::oneapi::boo>());
  static_assert(P2::has_property<sycl::property::buffer::use_host_ptr>());
  static_assert(P2::has_property<sycl::property::image::context_bound>());

  // Check has_property for a mix properties
  using P3 =
      sycl::ext::oneapi::property_list_t<sycl::property::buffer::use_host_ptr,
                                         sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::property::image::context_bound>;
  static_assert(!P3::has_property<sycl::ext::oneapi::bar>());
  static_assert(P3::has_property<sycl::ext::oneapi::baz>());
  static_assert(!P3::has_property<sycl::ext::oneapi::boo>());
  static_assert(P3::has_property<sycl::property::buffer::use_host_ptr>());
  static_assert(P3::has_property<sycl::property::image::context_bound>());
}
