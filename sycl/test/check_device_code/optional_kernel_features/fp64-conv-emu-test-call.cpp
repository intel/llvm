// UNSUPPORTED: cuda, hip

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-targets=spir64_gen -fsycl-fp64-conv-emu %s -c -S -emit-llvm -o- | FileCheck %s

// CHECK: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(){{.*}} !sycl_used_aspects ![[ASPECTFP64:[0-9]+]]
// CHECK-DAG: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlvE_(){{.*}} !sycl_used_aspects ![[ASPECTFP64]]
// CHECK-DAG: ![[ASPECTFP64]] = !{![[TMP:[0-9]+]]}
// CHECK-DAG: ![[TMP]] = !{!"fp64", i32 6}

// Tests if -fsycl-fp64-conv-emu option helps to correctly generate fp64 aspect.

#include <sycl/sycl.hpp>

struct T1 {
  double x;
};
struct T2 {
  struct T1 y;
};

SYCL_EXTERNAL double foo(double a);
double bar_compute(double a) { return a + 1.0; }

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      double a[10];
      double b[10];
      struct T2 tmp;
      int i = 4;
      b[i] = foo(a[i]);
      tmp.y.x = foo(a[i]);
    });
  });
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      double a[10];
      double b[10];
      int i = 4;
      b[i] = bar_compute(a[i]);
    });
  });
  return 0;
}
