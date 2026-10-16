// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the khr stream constructor rejects khr properties that are not
// stream properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

void foreign_property(sycl::handler &cgh) {
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::stream'}}
  sycl::stream s1(1024, 256, cgh, kp::in_order{});
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::stream'}}
  sycl::stream s2(1024, 256, cgh, properties{kp::enable_profiling{}});
}
