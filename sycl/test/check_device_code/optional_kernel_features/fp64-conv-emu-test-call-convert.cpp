// UNSUPPORTED: cuda, hip

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-targets=spir64_gen -fsycl-fp64-conv-emu %s -c -S -emit-llvm -o- | FileCheck %s

// CHECK-NOT: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(){{.*}} !sycl_used_aspects

// Tests if -fsycl-fp64-conv-emu option helps to correctly generate fp64 aspect.

#include <sycl/sycl.hpp>

double bar_convert(double a) { return (double)((float)(a)); }

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      double a[10];
      double b[10];
      int i = 4;
      b[i] = bar_convert(a[i]);
    });
  });
  return 0;
}
