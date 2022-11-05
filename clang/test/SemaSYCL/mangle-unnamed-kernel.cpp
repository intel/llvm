// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda -triple spir64-unknown-unknown -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](sycl::handler &h) { h.single_task([] {}); });
  return 0;
}

// CHECK: _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_
// CHECK: _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlvE_
