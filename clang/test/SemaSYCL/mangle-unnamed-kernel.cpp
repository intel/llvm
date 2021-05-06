// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda -triple spir64-unknown-unknown-sycldevice -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) {
    h.single_task([] {});
    h.single_task([] {});
  });
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  return 0;
}

// Note: The first 'name' of each (the EE10000 and EE100001) is the left-most
// lambda with the reference capture. EACH are the declaration context of only 1
// lambda, which is the E10000 on the RHS, which is why their names look the
// same, they are 2 lambdas in different declaration contexts.
// You can see this differentiation in the 2nd check line below.
// CHECK: FunctionDecl {{.*}} _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE10000_clES2_EUlvE10000_
// CHECK: FunctionDecl {{.*}} _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE10000_clES2_EUlvE10001_
// CHECK: FunctionDecl {{.*}} _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE10001_clES2_EUlvE10000_
