// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/experimental/task_sequence_properties.hpp>

using namespace sycl::ext;

constexpr uint32_t TestResponseCapacity = 7;
constexpr uint32_t TestInvocationCapacity = 5;
int main() {
  // Check that is_property_key is correctly specialized
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::balanced_key>::value);
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::response_capacity_key>::value);
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::invocation_capacity_key>::value);

  // Check that is_property_value is correctly specialized
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::balanced)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::response_capacity<0>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::response_capacity<
                         TestResponseCapacity>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::invocation_capacity<0>)>::value);
  static_assert(oneapi::experimental::is_property_value<
                decltype(intel::experimental::invocation_capacity<
                         TestInvocationCapacity>)>::value);

  // Check that property lists will accept the new properties.
  using P = decltype(oneapi::experimental::properties(
      intel::experimental::balanced,
      intel::experimental::response_capacity<TestResponseCapacity>,
      intel::experimental::invocation_capacity<TestInvocationCapacity>));
  static_assert(oneapi::experimental::is_property_list_v<P>);
  static_assert(P::has_property<intel::experimental::balanced_key>());
  static_assert(P::has_property<intel::experimental::response_capacity_key>());
  static_assert(
      P::has_property<intel::experimental::invocation_capacity_key>());
  static_assert(P::get_property<intel::experimental::balanced_key>() ==
                intel::experimental::balanced);
  static_assert(P::get_property<intel::experimental::response_capacity_key>() ==
                intel::experimental::response_capacity<TestResponseCapacity>);
  static_assert(
      P::get_property<intel::experimental::invocation_capacity_key>() ==
      intel::experimental::invocation_capacity<TestInvocationCapacity>);
}