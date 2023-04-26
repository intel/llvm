// RUN: %clangxx -fsycl-device-only -fsycl-native-cpu -Xclang -fsycl-int-header=%t.h -Xclang -fsycl-int-footer=%t-footer.h %s -o %t.bc
// RUN: %clangxx -D __SYCL_NATIVE_CPU__ -std=c++17 -include %t.h -I %sycl_include -I %sycl_include/sycl  %s -O2 -c -o %t-host.o
// RUN: %clangxx %t.bc -O3 -c -o %t-kernel.o
// RUN: %clangxx -L %sycl_libs_dir -lsycl %t-kernel.o %t-host.o -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="host:*" %t

#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class SimpleVadd;

int main() {
  const size_t N = 4;
  std::array<int, N> A = {{1, 2, 3, 4}}, B = {{2, 3, 4, 5}}, C{{0, 0, 0, 0}};
  cl::sycl::queue deviceQueue;
  cl::sycl::range<1> numOfItems{N};
  cl::sycl::buffer<int, 1> bufferA(A.data(), numOfItems);
  cl::sycl::buffer<int, 1> bufferB(B.data(), numOfItems);
  cl::sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

  deviceQueue
      .submit([&](cl::sycl::handler &cgh) {
        auto accessorA = bufferA.get_access<sycl_read>(cgh);
        auto accessorB = bufferB.get_access<sycl_read>(cgh);
        auto accessorC = bufferC.get_access<sycl_write>(cgh);

        auto kern = [=](cl::sycl::id<1> wiID) {
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
