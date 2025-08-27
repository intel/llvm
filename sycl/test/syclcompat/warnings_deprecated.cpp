// Test to verify that syclcompat namespace and APIs generate deprecation
// warnings.

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -Wall -Wextra

#include <syclcompat/syclcompat.hpp>

int main() {
  // Test deprecated namespace
  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::dim3 grid(1, 1, 1);

  // expected-warning@+1{{'syclcompat' is deprecated}}
  auto queue = syclcompat::get_default_queue();

  // Test deprecated memory APIs
  // expected-warning@+1{{'syclcompat' is deprecated}}
  void *ptr = syclcompat::malloc(1024);

  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::free(ptr);

  // Test deprecated utility APIs
  // expected-warning@+1{{'syclcompat' is deprecated}}
  auto device_count = syclcompat::device_count();

  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::wait();

  // Test deprecated atomic APIs
  int value = 42;
  int operand = 10;
  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::atomic_fetch_add(&value, operand);

  // Test deprecated math APIs
  // expected-warning@+1{{'syclcompat' is deprecated}}
  auto result = syclcompat::max(1, 2);

  // Test deprecated device APIs
  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::device_info info;

  // Test deprecated experimental APIs
  // expected-warning@+1{{'syclcompat' is deprecated}}
  syclcompat::experimental::launch_policy my_config(
      sycl::nd_range<1>{{32}, {32}});

  return 0;
}
