// REQUIRES: level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: ze_debug
//==------- non-uniform-wk-gp-test.cpp -------==//
// This is a diagnostic test which verifies that
// for loops with non-uniform work groups size
// errors are handled correctly.
//==------------------------------------------==//

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/stream.hpp>

using namespace sycl;

int test() {
  queue q = queue();
  auto device = q.get_device();
  auto deviceName = device.get_info<sycl::info::device::name>();
  std::cout << " Device Name: " << deviceName << std::endl;

  int res = 1;
  try {
    const int N = 1;
    q.submit([&](handler &cgh) {
      sycl::stream kernelout(108 * 64 + 128, 64, cgh);
      cgh.parallel_for<class test_kernel>(
          nd_range<3>(range<3>{1, 1, N}, range<3>{1, 1, 16}),
          [=](nd_item<3> itm) {
            kernelout << "Coordinates: " << itm.get_global_id() << sycl::endl;
          });
    });
    std::cout << "Test failed: no exception thrown." << std::endl;
  } catch (exception &E) {
    if (E.code() == errc::nd_range &&
        std::string(E.what()).find(
            "Non-uniform work-groups are not supported by the target device") !=
            std::string::npos) {
      std::cout << E.what() << std::endl;
      std::cout << "Test passed: caught the expected error." << std::endl;
      res = 0;
    } else {
      std::cout << E.what() << std::endl;
      std::cout << "Test failed: received error is incorrect." << std::endl;
    }
  }
  q.wait();

  std::cout << "Test passed: results are correct." << std::endl;
  return res;
}

int main() {
  int pltCount = 0, ret = 0;
  for (const auto &plt : platform::get_platforms()) {
    std::cout << "Platform #" << pltCount++ << ":" << std::endl;
    if (plt.get_backend() == backend::ext_oneapi_level_zero) {
      std::cout << "Backend: Level Zero" << std::endl;
      ret += test();
    }
    std::cout << std::endl;
  }
  return pltCount > 0 ? ret : -1;
}
