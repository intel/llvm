// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.out
// RUN: env ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4 %{l0_leak_check} %{run} %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=1 ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=4 %{l0_leak_check} %{run} %t.out

// Check that events and pools are not leaked when event caching is
// enabled/disabled.

#include <array>
#include <sycl/detail/core.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

int main() {
  sycl::queue deviceQueue;

  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}}, B = {{1, 2, 3, 4}}, C;
  sycl::range<1> numOfItems{array_size};
  sycl::buffer<int, 1> bufferA(A.data(), numOfItems);
  sycl::buffer<int, 1> bufferB(B.data(), numOfItems);
  sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

  for (int i = 0; i < 256; i++) {
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto accessorA = bufferA.get_access<sycl_read>(cgh);
      auto accessorB = bufferB.get_access<sycl_read>(cgh);
      auto accessorC = bufferC.get_access<sycl_write>(cgh);

      cgh.parallel_for<class SimpleVadd>(numOfItems, [=](sycl::id<1> wiID) {
        accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
      });
    });
  }
  return 0;
}
