// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <stdlib.h>

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue q;

  unsigned long long *x = sycl::malloc_shared<unsigned long long>(1, q);

  int error = 0;

  constexpr int N_ITERS = 64;

  // test barrier without arguments
  *x = 0;
  for (int i = 0; i < N_ITERS; i++) {
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

  q.wait_and_throw();
  auto Check = [&]() {
    std::bitset<8 * sizeof(*x)> bits(*x);
    std::bitset<8 * sizeof(*x)> ref;
    for (int i = 0; i < N_ITERS; ++i)
      ref[i] = 1;
    std::cout << "got: " << bits << "\nref: " << ref << std::endl;
    error |= (bits != ref);
  };
  Check();

  // test barrier when events are passed arguments
  *x = 0;
  for (int i = 0; i < N_ITERS; i++) {
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
  Check();

  std::cout << (error ? "failed\n" : "passed\n");

  sycl::free(x, q);

  return error;
}
