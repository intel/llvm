// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

// CUDA backend currently generates invalid binaries for this
// XFAIL: cuda

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class TestClass1 {};
class TestClass2 {};

int main() {
  // Check only compile-time properties are device-copyable
  using P1 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::baz::value_t<1>,
      sycl::ext::oneapi::experimental::boo::value_t<TestClass1, TestClass2>,
      sycl::ext::oneapi::experimental::bar::value_t>;

  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::baz::value_t<1>>);
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::boo::value_t<
          TestClass1, TestClass2>>);
  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::bar::value_t>);
  static_assert(sycl::is_device_copyable_v<P1>);

  // Check property list with non-device-copyable property
  using P2 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::bar::value_t,
      sycl::ext::oneapi::experimental::foz>;
  static_assert(
      !sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foz>);
  static_assert(!sycl::is_device_copyable_v<P2>);

  // Check property list with device-copyable compile-time and runtime
  // properties
  using P3 = sycl::ext::oneapi::experimental::property_list_t<
      sycl::ext::oneapi::experimental::baz::value_t<1>,
      sycl::ext::oneapi::experimental::foo>;
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foo>);
  static_assert(sycl::is_device_copyable_v<P3>);

  // Check that device-copyable property list can indeed be used in a kernel
  const auto PropertyList = sycl::ext::oneapi::experimental::property_list(
      sycl::ext::oneapi::experimental::baz_v<1>,
      sycl::ext::oneapi::experimental::foo{0});

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() {
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::baz>();
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::foo>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::baz>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::foo>();
    });
  });
}
