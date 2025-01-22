// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/latency_control/properties.hpp>

using namespace sycl::ext;

int main() {
  // Check that oneapi::experimental::is_property_value is correctly specialized
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::latency_anchor_id<-1>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::latency_constraint<
                         0, intel::experimental::latency_control_type::none,
                         0>)>::value);

  // Check that property lists will accept the new properties
  using P = decltype(oneapi::experimental::properties(
      intel::experimental::latency_anchor_id<-1>,
      intel::experimental::latency_constraint<
          0, intel::experimental::latency_control_type::none, 0>));
  static_assert(oneapi::experimental::is_property_list_v<P>);
  static_assert(P::has_property<intel::experimental::latency_anchor_id_key>());
  static_assert(P::has_property<intel::experimental::latency_constraint_key>());
  static_assert(P::get_property<intel::experimental::latency_anchor_id_key>() ==
                intel::experimental::latency_anchor_id<-1>);
  static_assert(
      P::get_property<intel::experimental::latency_constraint_key>() ==
      intel::experimental::latency_constraint<
          0, intel::experimental::latency_control_type::none, 0>);
}
