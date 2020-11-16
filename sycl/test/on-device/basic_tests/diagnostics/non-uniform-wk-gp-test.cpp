// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: opencl
// REQUIRES: level_zero
//==------- non-uniform-wk-gp-test.cpp -------==//
// This is a diagnostic test which verifies that
// for loops with non-uniform work groups size
// errors are handled correctly.
//==------------------------------------------==//

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int test() {
  try {
    queue q = queue();
    auto device = q.get_device();
    auto deviceName = device.get_info<cl::sycl::info::device::name>();
    std::cout << " Device Name: " << deviceName << std::endl;

    const int N = 1;
    q.submit([&](handler &cgh) {
      cl::sycl::stream kernelout(108 * 64 + 128, 64, cgh);
      cgh.parallel_for<class test_kernel>(
          nd_range<3>(range<3>{1, 1, N}, range<3>{1, 1, 16}),
          [=](nd_item<3> itm) {
            kernelout << "Coordinates: " << itm.get_global_id()
                      << cl::sycl::endl;
          });
    });

  } catch (sycl::runtime_error &E) {
    if (std::string(E.what()).find(
            "Specified local size doesn't match the required work-group size "
            "specified in the program source") != std::string::npos) {
      std::cout << E.what() << std::endl;
      std::cout << "Test passed: caught the expected error." << std::endl;
      return 0;
    } else {
      std::cout << E.what() << std::endl;
      std::cout << "Test failed: received error is incorrect." << std::endl;
      return 1;
    }
  }

  std::cout << "Test passed: results are correct." << std::endl;
  return 0;
}

int main() {

  int pltCount = 0, ret;
  for (const auto &plt : platform::get_platforms()) {
    if (!plt.has(aspect::host)) {
      std::cout << "Platform #" << pltCount++ << ":" << std::endl;
      if (plt.get_backend() == backend::level_zero) {
        std::cout << "Backend: Level Zero" << std::endl;
        ret = test();
      }
    }
    std::cout << std::endl;
  }
  return 0;
}
