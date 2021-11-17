// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class TestClass1 {};
class TestClass2 {};

int main() {
  // Check only compile-time properties are device-copyable
  using P1 = sycl::ext::oneapi::property_list_t<
      sycl::ext::oneapi::baz::value_t<1>,
      sycl::ext::oneapi::boo::value_t<TestClass1, TestClass2>,
      sycl::ext::oneapi::bar::value_t>;

  static_assert(sycl::is_device_copyable_v<sycl::ext::oneapi::baz::value_t<1>>);
  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::boo::value_t<TestClass1, TestClass2>>);
  static_assert(sycl::is_device_copyable_v<sycl::ext::oneapi::bar::value_t>);
  static_assert(sycl::is_device_copyable_v<P1>);

  // Check property list with non-device-copyable property
  using P2 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::bar::value_t,
                                         sycl::property::buffer::context_bound>;
  static_assert(
      !sycl::is_device_copyable_v<sycl::property::buffer::context_bound>);
  static_assert(!sycl::is_device_copyable_v<P2>);

  // Check property list with device-copyable compile-time and runtime
  // properties
  using P3 =
      sycl::ext::oneapi::property_list_t<sycl::ext::oneapi::baz::value_t<1>,
                                         sycl::property::image::use_host_ptr>;
  static_assert(
      sycl::is_device_copyable_v<sycl::property::image::use_host_ptr>);
  static_assert(sycl::is_device_copyable_v<P3>);

  // Check that device-copyable property list can indeed be used in a kernel
  const auto PropertyList = sycl::ext::oneapi::property_list(
      sycl::ext::oneapi::baz_v<1>, sycl::property::image::use_host_ptr{});

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() {
      decltype(PropertyList)::has_property<sycl::ext::oneapi::baz>();
      decltype(
          PropertyList)::has_property<sycl::property::image::use_host_ptr>();
      PropertyList.get_property<sycl::ext::oneapi::baz>();
      PropertyList.get_property<sycl::property::image::use_host_ptr>();
    });
  });
}
