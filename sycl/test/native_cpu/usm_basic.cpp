// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

int main() {
  const size_t N = 4;
  std::array<int, N> A = {{1, 2, 3, 4}}, B = {{2, 3, 4, 5}},
                     C{{-1, -1, -1, -1}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  auto a_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto b_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto c_ptr = sycl::malloc_device<int>(N, deviceQueue);
  deviceQueue.memcpy(a_ptr, A.data(), N * sizeof(int)).wait();
  deviceQueue.memcpy(b_ptr, B.data(), N * sizeof(int)).wait();
  deviceQueue.memset(c_ptr, 0, N * sizeof(int)).wait();
  deviceQueue.memcpy(C.data(), c_ptr, N * sizeof(int)).wait();
  // check that memset worked ok
  for (unsigned i = 0; i < N; i++) {
    if (C[i] != 0) {
      std::cout << "Error with memset at element " << i << " is " << C[i]
                << "\n";
      return 1;
    }
  }

  deviceQueue
      .submit([&](sycl::handler &cgh) {
        auto kern = [=](sycl::id<1> wiID) {
          c_ptr[wiID] = a_ptr[wiID] + b_ptr[wiID];
        };
        cgh.parallel_for(numOfItems, kern);
      })
      .wait();
  deviceQueue.memcpy(C.data(), c_ptr, N * sizeof(int)).wait();

  for (unsigned int i = 0; i < N; i++) {
    std::cout << "C[" << i << "] = " << C[i] << "\n";
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
  }
  sycl::free(a_ptr, deviceQueue);
  sycl::free(b_ptr, deviceQueue);
  sycl::free(c_ptr, deviceQueue);
  std::cout << "The results are correct!\n";
  return 0;
}
