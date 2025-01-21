// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/experimental/pipe_properties.hpp>

using namespace sycl::ext;

constexpr sycl::ext::intel::experimental::protocol_name TestProtocol =
    sycl::ext::intel::experimental::protocol_name::avalon_streaming;

int main() {
  // Check that is_property_value is correctly specialized.
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::ready_latency<3>)>::value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::bits_per_symbol<3>)>::value);

  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::uses_valid<true>)>::value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::uses_valid_on)>::value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::uses_valid_off)>::value);

  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::intel::experimental::
                             first_symbol_in_high_order_bits<true>)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::intel::experimental::
                             first_symbol_in_high_order_bits_on)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::intel::experimental::
                             first_symbol_in_high_order_bits_off)>::value);

  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::protocol<TestProtocol>)>::
          value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::protocol_avalon_streaming)>::
          value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::intel::experimental::
                             protocol_avalon_streaming_uses_ready)>::value);
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<
          decltype(sycl::ext::intel::experimental::protocol_avalon_mm)>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<
                decltype(sycl::ext::intel::experimental::
                             protocol_avalon_mm_uses_ready)>::value);

  // Checks that fully specialized properties are the same as the templated
  // variants.
  static_assert(std::is_same_v<
                decltype(sycl::ext::intel::experimental::uses_valid_on),
                decltype(sycl::ext::intel::experimental::uses_valid<true>)>);
  static_assert(
      std::is_same_v<decltype(sycl::ext::intel::experimental::
                                  first_symbol_in_high_order_bits_on),
                     decltype(sycl::ext::intel::experimental::
                                  first_symbol_in_high_order_bits<true>)>);
  static_assert(
      std::is_same_v<
          decltype(sycl::ext::intel::experimental::protocol_avalon_streaming),
          decltype(sycl::ext::intel::experimental::protocol<TestProtocol>)>);
  static_assert(
      std::is_same_v<decltype(sycl::ext::intel::experimental::
                                  protocol_avalon_streaming_uses_ready),
                     decltype(sycl::ext::intel::experimental::protocol<
                              sycl::ext::intel::experimental::protocol_name::
                                  avalon_streaming_uses_ready>)>);
  static_assert(
      std::is_same_v<
          decltype(sycl::ext::intel::experimental::protocol_avalon_mm),
          decltype(sycl::ext::intel::experimental::protocol<
                   sycl::ext::intel::experimental::protocol_name::avalon_mm>)>);
  static_assert(
      std::is_same_v<decltype(sycl::ext::intel::experimental::
                                  protocol_avalon_mm_uses_ready),
                     decltype(sycl::ext::intel::experimental::protocol<
                              sycl::ext::intel::experimental::protocol_name::
                                  avalon_mm_uses_ready>)>);

  // Check that property lists will accept the new properties.
  using P = decltype(sycl::ext::oneapi::experimental::properties(
      sycl::ext::intel::experimental::ready_latency<1>,
      sycl::ext::intel::experimental::bits_per_symbol<2>,
      sycl::ext::intel::experimental::uses_valid<true>,
      sycl::ext::intel::experimental::first_symbol_in_high_order_bits_off,
      sycl::ext::intel::experimental::protocol_avalon_streaming));
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<P>);
  static_assert(
      P::has_property<sycl::ext::intel::experimental::ready_latency_key>());
  static_assert(
      P::has_property<sycl::ext::intel::experimental::bits_per_symbol_key>());
  static_assert(
      P::has_property<sycl::ext::intel::experimental::uses_valid_key>());
  static_assert(P::has_property<sycl::ext::intel::experimental::
                                    first_symbol_in_high_order_bits_key>());
  static_assert(
      P::has_property<sycl::ext::intel::experimental::protocol_key>());

  static_assert(
      P::get_property<sycl::ext::intel::experimental::ready_latency_key>() ==
      sycl::ext::intel::experimental::ready_latency<1>);
  static_assert(
      P::get_property<sycl::ext::intel::experimental::bits_per_symbol_key>() ==
      sycl::ext::intel::experimental::bits_per_symbol<2>);
  static_assert(
      P::get_property<sycl::ext::intel::experimental::uses_valid_key>() ==
      sycl::ext::intel::experimental::uses_valid<true>);
  static_assert(
      P::get_property<sycl::ext::intel::experimental::
                          first_symbol_in_high_order_bits_key>() ==
      sycl::ext::intel::experimental::first_symbol_in_high_order_bits_off);
  static_assert(
      P::get_property<sycl::ext::intel::experimental::protocol_key>() ==
      sycl::ext::intel::experimental::protocol_avalon_streaming);
}
