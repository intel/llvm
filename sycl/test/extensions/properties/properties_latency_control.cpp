// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

#include <sycl/ext/oneapi/latency_control/properties.hpp>

using namespace sycl::ext::oneapi::experimental;

int main() {
  // Check that is_property_key is correctly specialized.
  static_assert(is_property_key<latency_anchor_id_key>::value);
  static_assert(is_property_key<latency_constraint_key>::value);

  // Check that is_property_value is correctly specialized.
  static_assert(is_property_value<decltype(latency_anchor_id<-1>)>::value);
  static_assert(
      is_property_value<decltype(latency_constraint<
                                 0, latency_control_type::none, 0>)>::value);

  // Check that property lists will accept the new properties.
  using P = decltype(properties(
      latency_anchor_id<-1>,
      latency_constraint<0, latency_control_type::none, 0>));
  static_assert(is_property_list_v<P>);
  static_assert(P::has_property<latency_anchor_id_key>());
  static_assert(P::has_property<latency_constraint_key>());
  static_assert(P::get_property<latency_anchor_id_key>() ==
                latency_anchor_id<-1>);
  static_assert(P::get_property<latency_constraint_key>() ==
                latency_constraint<0, latency_control_type::none, 0>);
}