// UNSUPPORTED: cuda, hip

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-targets=spir64_gen -fsycl-fp64-conv-emu %s -c -S -emit-llvm -o- | FileCheck %s

// CHECK-NOT: define {{.*}} spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(){{.*}} !sycl_used_aspects

// Tests if -fsycl-fp64-conv-emu option helps to correctly generate fp64 aspect
// in the presence of possibly recursive type declaration.

#include <sycl/sycl.hpp>

struct T2;
struct T1 {
  struct T2 *x;
};
struct T2 {
  struct T1 *y;
};

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      struct T1 tmp1;
      struct T2 tmp2;
    });
  });
  return 0;
}
