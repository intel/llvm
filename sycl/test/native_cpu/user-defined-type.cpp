// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t
#include <functional>
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

// Checks that the integration header can be compiled during host compilation
// even when it contains user-defined types
using myint = int;
struct myfun {
  int *ptr1;
  int *ptr2;
  myint param;

  myfun(int *ptr1, int *ptr2, int param)
      : ptr1(ptr1), ptr2(ptr2), param(param) {}
  void operator()(sycl::id<1> id) const { ptr1[id] = ptr2[id] + param; }
};

int main() {
  const size_t N = 4;
  std::array<int, N> A = {{0, 0, 0, 0}}, B = {{1, 2, 3, 4}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  auto a_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto b_ptr = sycl::malloc_device<int>(N, deviceQueue);
  deviceQueue.memcpy(a_ptr, A.data(), N * sizeof(int)).wait();
  deviceQueue.memcpy(b_ptr, B.data(), N * sizeof(int)).wait();
  myint param = 1;
  myfun TheFun(a_ptr, b_ptr, param);

  deviceQueue
      .submit([&](sycl::handler &cgh) { cgh.parallel_for(numOfItems, TheFun); })
      .wait();
  deviceQueue.memcpy(A.data(), a_ptr, N * sizeof(int)).wait();

  for (unsigned int i = 0; i < N; i++) {
    if (A[i] != B[i] + param) {
      std::cout << "The results are incorrect (element " << i << " is " << A[i]
                << "!\n";
      return 1;
    }
  }
  sycl::free(a_ptr, deviceQueue);
  sycl::free(b_ptr, deviceQueue);
  std::cout << "The results are correct!\n";
  return 0;
}
