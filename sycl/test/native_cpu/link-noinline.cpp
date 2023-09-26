// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class Test;

static constexpr int DEVICE_RET = 1;
static constexpr int HOST_RET = 2;

__attribute__((noinline)) int get_val() {
#ifdef __SYCL_DEVICE_ONLY__
  return DEVICE_RET;
#else
  return HOST_RET;
#endif
}

int main() {
  const size_t N = 4;
  std::array<int, N> C{{0, 0, 0, 0}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

  deviceQueue
      .submit([&](sycl::handler &cgh) {
        auto accessorC = bufferC.get_access<sycl_write>(cgh);

        auto kern = [=](sycl::id<1> wiID) { accessorC[wiID] = get_val(); };
        cgh.parallel_for<class SimpleVadd>(numOfItems, kern);
      })
      .wait();

  for (unsigned int i = 0; i < N; i++) {
    if (C[i] != DEVICE_RET) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
