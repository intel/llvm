// RUN: %clang_cc1 -triple spir64-unknown-unknown -internal-isystem %S/Inputs \
// RUN: -complex-range=promoted -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"
#include "half_type.hpp"

extern "C" SYCL_EXTERNAL float sqrt(float);
extern "C" SYCL_EXTERNAL float sqrt_half(half);

using namespace sycl;

int main() {
  const unsigned array_size = 4;
  range<1> numOfItems{array_size};
  float Value1 = .5f;
  float Value2 = .9f;
  half HalfValue1;
  half HalfValue2;
  queue deviceQueue;
  float *a;

    deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelFdiv>(numOfItems,
    [=](id<1> wiID) {
      // CHECK: fdiv float %4, %5
      a[0] = Value1 / Value2;
    });
  });
}
