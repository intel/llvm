// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests the sycl_khr_properties buffer/image properties (use_host_ptr,
// use_mutex, context_bound): their traits and that a buffer can be constructed
// with them. (The properties are shared with the image classes; see
// image_properties.cpp.)

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/buffer.hpp>

#include <mutex>
#include <vector>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

using buf = sycl::buffer<int, 1>;

struct OtherClass {};

// Property/key traits: all three are runtime properties.
static_assert(is_property_v<kp::use_host_ptr> && is_property_v<kp::use_mutex> &&
              is_property_v<kp::context_bound>);
static_assert(is_property_key_v<kp::key::use_host_ptr> &&
              is_property_key_v<kp::key::use_mutex> &&
              is_property_key_v<kp::key::context_bound>);
static_assert(!is_property_key_compile_time_v<kp::key::use_host_ptr> &&
              !is_property_key_compile_time_v<kp::key::use_mutex> &&
              !is_property_key_compile_time_v<kp::key::context_bound>);

// Registered for buffer, not for arbitrary classes.
static_assert(is_property_for_v<kp::use_host_ptr, buf> &&
              is_property_for_v<kp::use_mutex, buf> &&
              is_property_for_v<kp::context_bound, buf>);
static_assert(!is_property_for_v<kp::use_host_ptr, OtherClass> &&
              !is_property_for_v<kp::use_mutex, OtherClass> &&
              !is_property_for_v<kp::context_bound, OtherClass>);

void ctors(std::mutex &m, sycl::context ctx) {
  sycl::range<1> r{4};
  std::vector<int> v(4);

  // Single property and property list.
  sycl::buffer<int, 1> b1{r, kp::use_host_ptr{true}};
  sycl::buffer<int, 1> b2{v,
                          properties{kp::use_mutex{m}, kp::context_bound{ctx}}};

  // Old-style construction must still resolve unambiguously.
  sycl::buffer<int, 1> o1{r};
  sycl::buffer<int, 1> o2{r, sycl::property_list{}};
  sycl::buffer<int, 1> o3{v};
  (void)b1;
  (void)b2;
  (void)o1;
  (void)o2;
  (void)o3;
}
