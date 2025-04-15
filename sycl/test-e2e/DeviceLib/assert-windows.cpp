// REQUIRES: windows
// XFAIL: opencl && gpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364
//
// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out
//
// RUN: not env SHOULD_CRASH=1 SYCL_DEVICELIB_INHIBIT_NATIVE=1 CL_CONFIG_USE_VECTORIZER=False \
// RUN: %{run} %t.out 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-MESSAGE
//
// CHECK-MESSAGE: {{.*}}assert-windows.cpp:{{[0-9]+}}: (null): global id:
// [{{[0-3]}},0,0], local id: [{{[0-3]}},0,0] Assertion `accessorC[wiID] == 0 &&
// "Invalid value"` failed.

#include "../helpers.hpp"
#include <array>
#include <assert.h>
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr auto sycl_read = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N> &VA, const std::array<T, N> &VB,
                 std::array<T, N> &VC) {
  queue deviceQueue([](sycl::exception_list ExceptionList) {
    for (std::exception_ptr ExceptionPtr : ExceptionList) {
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (sycl::exception &E) {
        std::cerr << E.what() << std::endl;
      } catch (...) {
        std::cerr << "Unknown async exception was caught." << std::endl;
      }
    }
  });

  bool shouldCrash = env::isDefined("SHOULD_CRASH");

  sycl::range<1> numOfItems{N};
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVaddT>(numOfItems, [=](sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
      if (shouldCrash) {
        assert(accessorC[wiID] == 0 && "Invalid value");
      }
    });
  });
  deviceQueue.wait_and_throw();
}

int main() {
  std::array<int, 3> A = {1, 2, 3};
  std::array<int, 3> B = {1, 2, 3};
  std::array<int, 3> C = {0, 0, 0};

  simple_vadd(A, B, C);
  return EXIT_SUCCESS;
}
