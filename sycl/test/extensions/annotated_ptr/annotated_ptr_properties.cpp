// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

static annotated_ptr<int, decltype(properties())> AnnotatedPtr1;
static annotated_ptr<int, decltype(properties(register_map))> AnnotatedPtr2;
static annotated_ptr<int, decltype(properties(conduit, stable))> AnnotatedPtr3;
static annotated_ptr<int, decltype(properties(buffer_location<1>,
                                              read_write_mode_read, stable,
                                              conduit))>
    AnnotatedPtr4;

struct A {};

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<register_map_key, T>::value);
  static_assert(is_property_key_of<conduit_key, T>::value);
  static_assert(is_property_key_of<stable_key, T>::value);

  static_assert(is_property_key_of<buffer_location_key, T>::value);
  static_assert(is_property_key_of<awidth_key, T>::value);
  static_assert(is_property_key_of<dwidth_key, T>::value);
  static_assert(is_property_key_of<latency_key, T>::value);
  static_assert(is_property_key_of<read_write_mode_key, T>::value);
  static_assert(is_property_key_of<maxburst_key, T>::value);
  static_assert(is_property_key_of<wait_request_key, T>::value);

  static_assert(is_property_value_of<decltype(register_map), T>::value);
  static_assert(is_property_value_of<decltype(conduit), T>::value);
  static_assert(is_property_value_of<decltype(stable), T>::value);

  static_assert(is_property_value_of<decltype(buffer_location<1>), T>::value);
  static_assert(is_property_value_of<decltype(awidth<2>), T>::value);
  static_assert(is_property_value_of<decltype(dwidth<8>), T>::value);
  static_assert(is_property_value_of<decltype(latency<0>), T>::value);
  static_assert(is_property_value_of<decltype(read_write_mode_read), T>::value);
  static_assert(is_property_value_of<decltype(maxburst<1>), T>::value);
  static_assert(
      is_property_value_of<decltype(wait_request_requested), T>::value);
}

// Checks is_property_key_of and is_property_value_of are false for non-pointer
// type T.
template <typename T> void checkIsValidPropertyOfNonPtr() {
  static_assert(
      is_valid_property<T, decltype(wait_request_not_requested)>::value ==
      false);
  static_assert(is_valid_property<T, decltype(latency<8>)>::value == false);
}

int main() {
  static_assert(is_property_key<register_map_key>::value);
  static_assert(is_property_key<buffer_location_key>::value);

  checkIsPropertyOf<decltype(AnnotatedPtr1)>();
  static_assert(!AnnotatedPtr1.has_property<register_map_key>());
  static_assert(!AnnotatedPtr1.has_property<buffer_location_key>());

  checkIsPropertyOf<decltype(AnnotatedPtr2)>();
  static_assert(AnnotatedPtr2.has_property<register_map_key>());
  static_assert(!AnnotatedPtr2.has_property<conduit_key>());
  static_assert(!AnnotatedPtr2.has_property<buffer_location_key>());
  static_assert(AnnotatedPtr2.get_property<register_map_key>() == register_map);

  checkIsPropertyOf<decltype(AnnotatedPtr3)>();
  static_assert(!AnnotatedPtr3.has_property<register_map_key>());
  static_assert(AnnotatedPtr3.has_property<conduit_key>());
  static_assert(AnnotatedPtr3.has_property<stable_key>());
  static_assert(!AnnotatedPtr3.has_property<buffer_location_key>());
  static_assert(AnnotatedPtr3.get_property<stable_key>() == stable);
  static_assert(AnnotatedPtr3.get_property<conduit_key>() == conduit);

  checkIsPropertyOf<decltype(AnnotatedPtr4)>();
  static_assert(!AnnotatedPtr4.has_property<register_map_key>());
  static_assert(AnnotatedPtr4.has_property<conduit_key>());
  static_assert(AnnotatedPtr4.has_property<stable_key>());
  static_assert(AnnotatedPtr4.has_property<buffer_location_key>());
  static_assert(AnnotatedPtr4.has_property<read_write_mode_key>());
  static_assert(AnnotatedPtr4.get_property<conduit_key>() == conduit);
  static_assert(AnnotatedPtr4.get_property<stable_key>() == stable);
  static_assert(AnnotatedPtr4.get_property<buffer_location_key>() ==
                buffer_location<1>);
  static_assert(AnnotatedPtr4.get_property<read_write_mode_key>() ==
                read_write_mode_read);

  // Check if a property is valid for a given type
  checkIsValidPropertyOfNonPtr<A>();

  // expected-error-re@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* {{static assertion failed due to requirement {{.+}}: FPGA Interface properties (i.e. awidth, dwidth, etc.) can only be set with BufferLocation together.}}
  annotated_ptr<int, decltype(properties(read_write_mode_read))> AnnotatedPtr5;
  return 0;
}
