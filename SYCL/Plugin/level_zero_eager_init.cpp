// REQUIRES: level_zero, level_zero_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZE_DEBUG=1 SYCL_EAGER_INIT=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// The test is to check that under SYCL_EAGER_INIT=1 there is no calls to
// heavy L0 initialization in the hot reportable path.
//
// CHECK-LABEL: HOT HOT HOT
// CHECK-NOT: zeCommandQueueCreate
// CHECK-NOT: zeCommandListCreate
// CHECK-NOT: zeFenceCreate
//

#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T> class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(cl::sycl::queue &Queue, const std::array<T, N> &VA,
                 const std::array<T, N> &VB, std::array<T, N> &VC) {
  cl::sycl::range<1> numOfItems{N};
  cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  Queue.submit([&](cl::sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(
        numOfItems, [=](cl::sycl::id<1> wiID) {
          accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
        });
  });
}

int main() {
  const size_t array_size = 4;
  std::array<cl::sycl::cl_int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  cl::sycl::queue Q;

  // simple_vadd(Q, A, B, C);
  std::cerr << "\n\n\nHOT HOT HOT\n\n\n";
  simple_vadd(Q, A, B, C);
  return 0;
}
