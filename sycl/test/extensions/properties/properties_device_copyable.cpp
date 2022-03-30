// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include "mock_compile_time_properties.hpp"

class TestClass1 {};
class TestClass2 {};

namespace sycl {
// Explicitly mark fir property as device-copyable
template <>
struct is_device_copyable<sycl::ext::oneapi::experimental::fir>
    : std::true_type {};
template <>
struct is_device_copyable<const sycl::ext::oneapi::experimental::fir>
    : std::true_type {};
} // namespace sycl

int main() {
  // Check only compile-time properties are device-copyable
  using P1 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<TestClass1, TestClass2>,
      sycl::ext::oneapi::experimental::bar));
  using CP1 = const P1;
  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::baz_key::value_t<1>>);
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::boo_key::
                                     value_t<TestClass1, TestClass2>>);
  static_assert(sycl::is_device_copyable_v<
                sycl::ext::oneapi::experimental::bar_key::value_t>);
  static_assert(sycl::is_device_copyable_v<P1>);
  static_assert(sycl::is_device_copyable_v<CP1>);

  // Check property list with non-device-copyable property
  using P2 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foz{42.42, false}));
  using CP2 = const P2;
  static_assert(
      !sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foz>);
  static_assert(!sycl::is_device_copyable_v<P2>);
  static_assert(!sycl::is_device_copyable_v<CP2>);

  // Check property list with explicit device-copyable property
  using P3 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::fir{42.42, false}));
  using CP3 = const P3;
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::fir>);
  static_assert(sycl::is_device_copyable_v<P3>);
  static_assert(sycl::is_device_copyable_v<CP3>);

  // Check property list with device-copyable compile-time and runtime
  // properties
  using P4 = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::foo{1234}));
  using CP4 = const P4;
  static_assert(
      sycl::is_device_copyable_v<sycl::ext::oneapi::experimental::foo>);
  static_assert(sycl::is_device_copyable_v<P4>);
  static_assert(sycl::is_device_copyable_v<CP4>);

  // Check that device-copyable property list can indeed be used in a kernel
  const auto PropertyList = sycl::ext::oneapi::experimental::properties(
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::foo{0},
      sycl::ext::oneapi::experimental::fir{12.34, true});

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task([=]() {
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::baz_key>();
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::foo_key>();
      decltype(PropertyList)::has_property<
          sycl::ext::oneapi::experimental::fir_key>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::baz_key>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::foo_key>();
      PropertyList.get_property<sycl::ext::oneapi::experimental::fir_key>();
    });
  });
  return 0;
}
