// RUN: %clang_cc1 -fsycl -fno-offload-fp32-prec-div -fsycl-use-spirv-backend-for-spirv-gen %s -o -
// Test that codegen doesn't crash.

#include "Inputs/sycl.hpp"

using namespace sycl;

int main() {
  const unsigned array_size = 4;
  range<1> numOfItems{array_size};
  queue deviceQueue;
  float *a;

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelFdiv>(numOfItems,
    [=](id<1> wiID) {
      a[0] = .5f / .9f;
    });
  });

  return 0;
}
