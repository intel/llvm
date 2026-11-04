// REQUIRES: linux, gpu && offload
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s

// nd_range launch with several arguments: goes through the intercepted
// zeKernelSetGroupSize + zeCommandListAppendLaunchKernel path.
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data + 1; }

class MyKernel;

int main() {
  sycl::queue Q;
  constexpr size_t N = 16;
  auto *A = sycl::malloc_device<int>(N, Q);
  auto *B = sycl::malloc_device<int>(N, Q);
  auto *C = sycl::malloc_device<int>(N, Q);
  auto *Out = sycl::malloc_device<int>(N, Q);
  Q.memset(A, 0, N * sizeof(int));
  Q.memset(B, 0, N * sizeof(int));
  Q.wait();

  Q.parallel_for<MyKernel>(sycl::nd_range<1>(N, 8), [=](sycl::nd_item<1> It) {
     size_t I = It.get_global_id(0);
     int V = A[I] + B[I];
     if (I == 13)
       V += C[I];
     Out[I] = check(V);
   }).wait();
  // CHECK: use-of-uninitialized-value
  // CHECK-NEXT: kernel <{{.*MyKernel}}> LID(5, 0, 0) GID(13, 0, 0)

  sycl::free(A, Q);
  sycl::free(B, Q);
  sycl::free(C, Q);
  sycl::free(Out, Q);
  return 0;
}
