// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class TestClass1 {};
class TestClass2 {};

int main() {
  // Check only compile-time properties are device-copyable
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<TestClass1, TestClass2>,
      sycl::ext::oneapi::experimental::bar));

  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::baz_key::value_t<1>>);
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::boo_key::
                                     value_t<TestClass1, TestClass2>>);
  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::bar_key::value_t>);
  static_assert(sycl::is_device_copyable_v<P1>);

  // Check property list with non-device-copyable property
  using P2 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foz{42.42, false}));
  static_assert(
      !sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foz>);
  static_assert(!sycl::is_device_copyable_v<P2>);

  // Check property list with device-copyable compile-time and runtime
  // properties
  using P3 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::foo{1234}));
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foo>);
  static_assert(sycl::is_device_copyable_v<P3>);

  // Check that device-copyable property list can indeed be used in a kernel
  const auto PropertyList = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::foo{0});

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() {
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::baz_key>();
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::foo_key>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::baz_key>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::foo_key>();
    });
  });
  return 0;
}
