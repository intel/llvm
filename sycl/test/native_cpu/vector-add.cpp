// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

// Same test but with -g
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -g -o %t-debug
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t-debug

// Test with vector width set manually, this ensures that we peel correctly when doing
// vectorization.
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -mllvm -sycl-native-cpu-vecz-width=4 %s -g -o %t-vec
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t-vec

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class SimpleVadd;

int main() {
  const size_t N = 5;
  std::array<int, N> A = {{1, 2, 3, 4, 5}}, B = {{2, 3, 4, 5, 6}},
    C{{0, 0, 0, 0, 0}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  sycl::buffer<int, 1> bufferA(A.data(), numOfItems);
  sycl::buffer<int, 1> bufferB(B.data(), numOfItems);
  sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

  deviceQueue
      .submit([&](sycl::handler &cgh) {
        auto accessorA = bufferA.get_access<sycl_read>(cgh);
        auto accessorB = bufferB.get_access<sycl_read>(cgh);
        auto accessorC = bufferC.get_access<sycl_write>(cgh);

        auto kern = [=](sycl::id<1> wiID) {
          accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
        };
        cgh.parallel_for<class SimpleVadd>(numOfItems, kern);
      })
      .wait();

  for (unsigned int i = 0; i < N; i++) {
    std::cout << "C[" << i << "] = " << C[i] << "\n";
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
