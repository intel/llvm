// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/experimental/fpga_kernel_properties.hpp>

using namespace sycl::ext;

int main() {
  // Check that oneapi::experimental::is_property_key is correctly specialized
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::streaming_interface_key>::value);
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::register_map_interface_key>::value);
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::pipelined_key>::value);

  // Check that oneapi::experimental::is_property_value is correctly specialized
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::streaming_interface<
                         intel::experimental::streaming_interface_options_enum::
                             accept_downstream_stall>)>::value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::
                       streaming_interface_accept_downstream_stall)>::value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::
                       streaming_interface_remove_downstream_stall)>::value);

  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::register_map_interface<
                   intel::experimental::register_map_interface_options_enum::
                       wait_for_done_write>)>::value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::
                       register_map_interface_wait_for_done_write)>::value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::
                       register_map_interface_do_not_wait_for_done_write)>::
          value);

  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::pipelined<-1>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::pipelined<0>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::pipelined<4>)>::value);

  // Check that property lists will accept the new properties
  using PS = decltype(oneapi::experimental::properties(
      intel::experimental::streaming_interface_remove_downstream_stall));
  static_assert(oneapi::experimental::is_property_list_v<PS>);
  static_assert(
      PS::has_property<intel::experimental::streaming_interface_key>());
  static_assert(
      PS::get_property<intel::experimental::streaming_interface_key>() ==
      intel::experimental::streaming_interface_remove_downstream_stall);

  using PR = decltype(oneapi::experimental::properties(
      intel::experimental::register_map_interface_wait_for_done_write));
  static_assert(oneapi::experimental::is_property_list_v<PR>);
  static_assert(
      PR::has_property<intel::experimental::register_map_interface_key>());
  static_assert(
      PR::get_property<intel::experimental::register_map_interface_key>() ==
      intel::experimental::register_map_interface_wait_for_done_write);

  using PP = decltype(oneapi::experimental::properties(
      intel::experimental::pipelined<-1>));
  static_assert(oneapi::experimental::is_property_list_v<PP>);
  static_assert(PP::has_property<intel::experimental::pipelined_key>());
  static_assert(PP::get_property<intel::experimental::pipelined_key>() ==
                intel::experimental::pipelined<-1>);
}
