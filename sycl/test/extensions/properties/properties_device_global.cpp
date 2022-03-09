// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include <sycl/ext/oneapi/device_global/properties.hpp>

using namespace sycl::ext::oneapi::experimental;

constexpr host_access_enum TestAccess = host_access_enum::read_write;
constexpr init_mode_enum TestTrigger = init_mode_enum::reset;

int main() {
  // Check that is_property_key is correctly specialized.
  static_assert(is_property_key<device_image_scope_key>::value);
  static_assert(is_property_key<host_access_key>::value);
  static_assert(is_property_key<init_mode_key>::value);
  static_assert(is_property_key<implement_in_csr_key>::value);

  // Check that is_property_value is correctly specialized.
  static_assert(is_property_value<decltype(device_image_scope)>::value);
  static_assert(is_property_value<decltype(host_access<TestAccess>)>::value);
  static_assert(is_property_value<decltype(host_access_read)>::value);
  static_assert(is_property_value<decltype(host_access_write)>::value);
  static_assert(is_property_value<decltype(host_access_read_write)>::value);
  static_assert(is_property_value<decltype(host_access_none)>::value);
  static_assert(is_property_value<decltype(init_mode<TestTrigger>)>::value);
  static_assert(is_property_value<decltype(init_mode_reprogram)>::value);
  static_assert(is_property_value<decltype(init_mode_reset)>::value);
  static_assert(is_property_value<decltype(implement_in_csr<true>)>::value);
  static_assert(is_property_value<decltype(implement_in_csr_on)>::value);
  static_assert(is_property_value<decltype(implement_in_csr_off)>::value);

  // Checks that fully specialized properties are the same as the templated
  // variants.
  static_assert(std::is_same_v<decltype(host_access_read),
                               decltype(host_access<host_access_enum::read>)>);
  static_assert(std::is_same_v<decltype(host_access_write),
                               decltype(host_access<host_access_enum::write>)>);
  static_assert(
      std::is_same_v<decltype(host_access_read_write),
                     decltype(host_access<host_access_enum::read_write>)>);
  static_assert(std::is_same_v<decltype(host_access_none),
                               decltype(host_access<host_access_enum::none>)>);
  static_assert(std::is_same_v<decltype(init_mode_reprogram),
                               decltype(init_mode<init_mode_enum::reprogram>)>);
  static_assert(std::is_same_v<decltype(init_mode_reset),
                               decltype(init_mode<init_mode_enum::reset>)>);
  static_assert(std::is_same_v<decltype(implement_in_csr_on),
                               decltype(implement_in_csr<true>)>);
  static_assert(std::is_same_v<decltype(implement_in_csr_off),
                               decltype(implement_in_csr<false>)>);

  // Check that property lists will accept the new properties.
  using P =
      decltype(properties(device_image_scope, host_access<TestAccess>,
                          init_mode<TestTrigger>, implement_in_csr<true>));
  static_assert(is_property_list_v<P>);
  static_assert(P::has_property<device_image_scope_key>());
  static_assert(P::has_property<host_access_key>());
  static_assert(P::has_property<init_mode_key>());
  static_assert(P::has_property<implement_in_csr_key>());
  static_assert(P::get_property<device_image_scope_key>() ==
                device_image_scope);
  static_assert(P::get_property<host_access_key>() == host_access<TestAccess>);
  static_assert(P::get_property<init_mode_key>() == init_mode<TestTrigger>);
  static_assert(P::get_property<implement_in_csr_key>() ==
                implement_in_csr<true>);
}
