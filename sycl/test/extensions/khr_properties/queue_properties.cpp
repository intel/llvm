// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests the sycl_khr_properties queue properties (enable_profiling, in_order):
// their traits, that a queue can be constructed with them, and the queue
// members is_in_order / khr_is_profiling_enabled.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct OtherClass {};

// Traits: both are runtime queue properties.
static_assert(is_property_v<kp::enable_profiling> &&
              is_property_v<kp::in_order>);
static_assert(is_property_key_v<kp::key::enable_profiling> &&
              is_property_key_v<kp::key::in_order>);
static_assert(!is_property_key_compile_time_v<kp::key::enable_profiling> &&
              !is_property_key_compile_time_v<kp::key::in_order>);
static_assert(is_property_for_v<kp::enable_profiling, sycl::queue> &&
              is_property_for_v<kp::in_order, sycl::queue>);
static_assert(!is_property_for_v<kp::enable_profiling, OtherClass>);

// A property list is not itself a property.
static_assert(!is_property_v<properties<kp::enable_profiling>>);
static_assert(is_property_list_for_v<
              decltype(properties{kp::enable_profiling{true}, kp::in_order{}}),
              sycl::queue>);

// A queue can be constructed from a single khr property or a khr property list.
void ctors() {
  sycl::queue q1{kp::enable_profiling{true}};
  sycl::queue q2{kp::in_order{false}};
  sycl::queue q3{properties{kp::enable_profiling{true}, kp::in_order{true}}};
  (void)q1.is_in_order();
  (void)q1.khr_is_profiling_enabled();
}
