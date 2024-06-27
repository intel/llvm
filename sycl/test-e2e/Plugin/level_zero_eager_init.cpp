// REQUIRES: level_zero, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 SYCL_EAGER_INIT=1 %{run} %t.out 2>&1 | FileCheck %s
//
// The test is to check that under SYCL_EAGER_INIT=1 there is no calls to
// heavy L0 initialization in the hot reportable path.
//
// CHECK-LABEL: HOT HOT HOT
// CHECK-NOT: ZE ---> zeCommandQueueCreate
// CHECK-NOT: ZE ---> zeCommandListCreate
// CHECK-NOT: ZE ---> zeFenceCreate
//

#include <sycl/detail/core.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T> class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(sycl::queue &Queue, const std::array<T, N> &VA,
                 const std::array<T, N> &VB, std::array<T, N> &VC) {
  sycl::range<1> numOfItems{N};
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  Queue.submit([&](sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, [=](sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    });
  });
}

int main() {
  const size_t array_size = 4;
  std::array<sycl::cl_int, array_size> A = {{1, 2, 3, 4}}, B = {{1, 2, 3, 4}},
                                       C;
  sycl::queue Q;

  // simple_vadd(Q, A, B, C);
  std::cerr << "\n\n\nHOT HOT HOT\n\n\n";
  simple_vadd(Q, A, B, C);
  return 0;
}
