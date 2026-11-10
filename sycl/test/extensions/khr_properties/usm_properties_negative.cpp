// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the khr USM allocation functions and usm_allocator constructors
// reject khr properties that are not USM properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

namespace kp = sycl::khr::property;
using namespace sycl::khr;
using sycl::usm::alloc;

void foreign_property(const sycl::queue &q, const sycl::device &dev,
                      const sycl::context &ctx) {
  properties props{kp::in_order{}, kp::enable_profiling{}};
  // expected-error@+1 {{no matching function for call to 'malloc_device'}}
  sycl::malloc_device(16, dev, ctx, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'malloc_device'}}
  sycl::malloc_device<int>(4, q, props);
  // expected-error@+1 {{no matching function for call to 'aligned_alloc_device'}}
  sycl::aligned_alloc_device(64, 16, q, kp::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'malloc_host'}}
  sycl::malloc_host(16, ctx, props);
  // expected-error@+1 {{no matching function for call to 'malloc_host'}}
  sycl::malloc_host<int>(4, q, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'aligned_alloc_host'}}
  sycl::aligned_alloc_host<int>(64, 4, ctx, props);
  // expected-error@+1 {{no matching function for call to 'malloc_shared'}}
  sycl::malloc_shared(16, q, props);
  // expected-error@+1 {{no matching function for call to 'malloc_shared'}}
  sycl::malloc_shared<int>(4, dev, ctx, kp::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'aligned_alloc_shared'}}
  sycl::aligned_alloc_shared(64, 16, dev, ctx, properties{kp::in_order{}});
  // expected-error@+1 {{no matching function for call to 'malloc'}}
  sycl::malloc(16, dev, ctx, alloc::device, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'malloc'}}
  sycl::malloc<int>(4, q, alloc::host, props);
  // expected-error@+1 {{no matching function for call to 'aligned_alloc'}}
  sycl::aligned_alloc(64, 16, q, alloc::shared, kp::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'aligned_alloc'}}
  sycl::aligned_alloc<int>(64, 4, dev, ctx, alloc::shared, props);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::usm_allocator<int, alloc::shared>'}}
  sycl::usm_allocator<int, alloc::shared> a1(ctx, dev, props);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::usm_allocator<int, alloc::host>'}}
  sycl::usm_allocator<int, alloc::host> a2(q, kp::in_order{});
}
