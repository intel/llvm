// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the reduction() factory functions reject khr properties that are
// not reduction properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

void foreign_property(int *usm) {
  // expected-error@+1 {{no matching function for call to 'reduction'}}
  auto r1 = sycl::reduction(usm, sycl::plus<int>(), kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'reduction'}}
  auto r2 = sycl::reduction(
      usm, 0, sycl::plus<int>(),
      properties{kp::initialize_to_identity{}, kp::enable_profiling{}});
  (void)r1;
  (void)r2;
}
