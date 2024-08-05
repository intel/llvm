// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -ferror-limit=0 \
// RUN: -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using annotated_ptr_load1 = annotated_ptr<
    float, decltype(properties(
               read_hint<cache_control<cache_mode::uncached, cache_level::L3,
                                       cache_level::L3>>))>;
using annotated_ptr_load2 = annotated_ptr<
    float,
    decltype(properties(
        read_hint<cache_control<cache_mode::cached, cache_level::L3>,
                  cache_control<cache_mode::uncached, cache_level::L3>>))>;
using annotated_ptr_load3 = annotated_ptr<
    float,
    decltype(properties(
        read_hint<cache_control<cache_mode::write_back, cache_level::L3>>))>;
using annotated_ptr_load4 =
    annotated_ptr<float,
                  decltype(properties(
                      read_assertion<cache_control<cache_mode::write_through,
                                                   cache_level::L3>>))>;
using annotated_ptr_load5 = annotated_ptr<
    float,
    decltype(properties(
        write_hint<cache_control<cache_mode::invalidate, cache_level::L3>>))>;

void cache_control_read_func(queue q) {
  float *ArrayA = malloc_shared<float>(10, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      // expected-error@sycl/ext/intel/experimental/cache_control_properties.hpp:* {{Duplicate cache_level L3 specification}}
      annotated_ptr_load1 src1{&ArrayA[0]};

      // expected-error@sycl/ext/intel/experimental/cache_control_properties.hpp:* {{Conflicting cache_mode at L3}}
      annotated_ptr_load2 src2{&ArrayA[0]};

      // expected-error@sycl/ext/intel/experimental/cache_control_properties.hpp:* {{read_hint must specify cache_mode uncached, cached or streaming}}
      annotated_ptr_load3 src3{&ArrayA[0]};

      // expected-error@sycl/ext/intel/experimental/cache_control_properties.hpp:* {{read_assertion must specify cache_mode invalidate or constant}}
      annotated_ptr_load4 src4{&ArrayA[0]};

      // expected-error@sycl/ext/intel/experimental/cache_control_properties.hpp:* {{write_hint must specify cache_mode uncached, write_through, write_back or streaming}}
      annotated_ptr_load5 src5{&ArrayA[0]};
    });
  });
}
