// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Temporarily disabled on HIP, CUDA and L0 due to sporadic failures.
// UNSUPPORTED: hip, level_zero, cuda

#include <iostream>
#include <stdlib.h>
#include <sycl/sycl.hpp>

int main() {
  sycl::device dev{sycl::default_selector_v};
  sycl::queue q{dev};

  unsigned long long *x = sycl::malloc_shared<unsigned long long>(1, q);

  int error = 0;

  // test barrier without arguments
  *x = 0;
  for (int i = 0; i < 64; i++) {
    q.single_task([=]() {
      // do some busywork
      volatile float y = *x;
      for (int j = 0; j < 100; j++) {
        y = sycl::cos(y);
      }
      // update the value
      *x *= 2;
    });
    q.ext_oneapi_submit_barrier();
    q.single_task([=]() { *x += 1; });
    q.ext_oneapi_submit_barrier();
  }

  std::cout << std::bitset<8 * sizeof(unsigned long long)>(*x) << std::endl;

  q.wait_and_throw();
  error |= (*x != (unsigned long long)-1);

  // test barrier when events are passed arguments
  *x = 0;
  for (int i = 0; i < 64; i++) {
    sycl::event e = q.single_task([=]() {
      // do some busywork
      volatile float y = *x;
      for (int j = 0; j < 100; j++) {
        y = sycl::cos(y);
      }
      // update the value
      *x *= 2;
    });
    q.ext_oneapi_submit_barrier({e});
    e = q.single_task([=]() { *x += 1; });
    q.ext_oneapi_submit_barrier({e});
  }

  q.wait_and_throw();
  error |= (*x != (unsigned long long)-1);

  std::cout << std::bitset<8 * sizeof(unsigned long long)>(*x) << std::endl;

  std::cout << (error ? "failed\n" : "passed\n");

  sycl::free(x, q);

  return error;
}
