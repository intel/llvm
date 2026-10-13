// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the khr context constructors reject khr properties that are not
// context properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

#include <vector>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

void foreign_property(const sycl::device &dev, const sycl::platform &plt,
                      const std::vector<sycl::device> &devs,
                      sycl::async_handler handler) {
  properties props{kp::in_order{}, kp::enable_profiling{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c1(kp::in_order{});
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c2(handler, props);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c3(dev, kp::enable_profiling{});
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c4(dev, handler, props);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c5(plt, kp::in_order{});
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c6(plt, handler, props);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c7(devs, properties{kp::in_order{}});
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::context'}}
  sycl::context c8(devs, handler, kp::enable_profiling{});
}
