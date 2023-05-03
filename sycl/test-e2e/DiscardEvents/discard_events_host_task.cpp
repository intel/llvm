// If necessary, the test can be removed as run_on_host_intel() is deprecated
// and host_task() which should be used instead does not use the PI call
// piEnqueueNativeKernel
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is `nullptr` for
// piEnqueueNativeKernel.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
//
// CHECK: ---> piEnqueueNativeKernel(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: The test passed.

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

void CheckArray(sycl::queue Q, int *x, size_t buffer_size, int expected) {
  Q.wait();
  for (size_t i = 0; i < buffer_size; ++i)
    assert(x[i] == expected);
}

static constexpr size_t BUFFER_SIZE = 16;

int main(int Argc, const char *Argv[]) {

  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(Props);

  int *x = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(x != nullptr);

  Q.submit([&](sycl::handler &CGH) {
    CGH.run_on_host_intel([=]() {
      for (size_t i = 0; i < BUFFER_SIZE; ++i)
        x[i] = 8;
    });
  });
  CheckArray(Q, x, BUFFER_SIZE, 8);

  Q.wait();
  free(x, Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
