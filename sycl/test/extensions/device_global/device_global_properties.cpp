// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// TODO: device_global currently requires device_image_scope. When this
// requirement is lifted the tests should include a case without any properties
// and DeviceGlobal2, DeviceGlobal3, and DeviceGlobal4 should have
// device_image_scope removed.
static device_global<int, decltype(properties(device_image_scope))>
    DeviceGlobal1;
static device_global<int,
                     decltype(properties(device_image_scope, host_access_none))>
    DeviceGlobal2;
static device_global<int, decltype(properties(device_image_scope))>
    DeviceGlobal3;
static device_global<int, decltype(properties(device_image_scope))>
    DeviceGlobal4;
static device_global<int, decltype(properties(host_access_write,
                                              device_image_scope))>
    DeviceGlobal5;

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<device_image_scope_key, T>::value);
  static_assert(is_property_key_of<host_access_key, T>::value);

  static_assert(is_property_value_of<decltype(device_image_scope), T>::value);
  static_assert(is_property_value_of<decltype(host_access_read), T>::value);
  static_assert(is_property_value_of<decltype(host_access_write), T>::value);
  static_assert(
      is_property_value_of<decltype(host_access_read_write), T>::value);
  static_assert(is_property_value_of<decltype(host_access_none), T>::value);
}

int main() {
  static_assert(is_property_value<decltype(device_image_scope)>::value);
  static_assert(is_property_value<decltype(host_access_read)>::value);
  static_assert(is_property_value<decltype(host_access_write)>::value);
  static_assert(is_property_value<decltype(host_access_read_write)>::value);
  static_assert(is_property_value<decltype(host_access_none)>::value);

  checkIsPropertyOf<decltype(DeviceGlobal1)>();
  static_assert(DeviceGlobal1.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal1.has_property<host_access_key>());

  checkIsPropertyOf<decltype(DeviceGlobal2)>();
  static_assert(DeviceGlobal2.has_property<device_image_scope_key>());
  static_assert(DeviceGlobal2.has_property<host_access_key>());
  static_assert(DeviceGlobal2.get_property<device_image_scope_key>() ==
                device_image_scope);
  static_assert(DeviceGlobal2.get_property<host_access_key>().value ==
                host_access_enum::none);

  checkIsPropertyOf<decltype(DeviceGlobal3)>();
  static_assert(DeviceGlobal3.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal3.has_property<host_access_key>());
  static_assert(DeviceGlobal3.get_property<device_image_scope_key>() ==
                device_image_scope);

  checkIsPropertyOf<decltype(DeviceGlobal4)>();
  static_assert(DeviceGlobal4.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal4.has_property<host_access_key>());
  static_assert(DeviceGlobal4.get_property<device_image_scope_key>() ==
                device_image_scope);

  checkIsPropertyOf<decltype(DeviceGlobal5)>();
  static_assert(DeviceGlobal5.has_property<device_image_scope_key>());
  static_assert(DeviceGlobal5.has_property<host_access_key>());
  static_assert(DeviceGlobal5.get_property<device_image_scope_key>() ==
                device_image_scope);
  static_assert(DeviceGlobal5.get_property<host_access_key>().value ==
                host_access_enum::write);

  return 0;
}
