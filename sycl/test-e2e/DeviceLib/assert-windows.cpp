// REQUIRES: cpu,windows
//
// FIXME: OpenCL CPU backend compiler crashes on a call to _wassert.
// Disable the test until the fix reaches SYCL test infrastructure.
// XFAIL: *
//
// RUN: %{build} -o %t.out
//
// MSVC implementation of assert does not call an unreachable built-in, so the
// program doesn't terminate when fallback is used.
//
// FIXME: SPIR-V Unreachable should be called from the fallback
// explicitly. Since the test is going to crash, we'll have to follow a similar
// approach as on Linux - call the test in a subprocess.
//
// RUN: env SYCL_UR_TRACE=2 SYCL_DEVICELIB_INHIBIT_NATIVE=1 CL_CONFIG_USE_VECTORIZER=False %{run} %t.out | FileCheck %s --check-prefix=CHECK-FALLBACK
// RUN: env SHOULD_CRASH=1 SYCL_DEVICELIB_INHIBIT_NATIVE=1 CL_CONFIG_USE_VECTORIZER=False %{run} %t.out | FileCheck %s --check-prefix=CHECK-MESSAGE
//
// CHECK-MESSAGE: {{.*}}assert-windows.cpp:{{[0-9]+}}: (null): global id:
// [{{[0-3]}},0,0], local id: [{{[0-3]}},0,0] Assertion `accessorC[wiID] == 0 &&
// "Invalid value"` failed.
//
// CHECK-FALLBACK: ---> urProgramLink

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
