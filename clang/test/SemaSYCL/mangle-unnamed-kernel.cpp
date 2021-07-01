// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda -triple spir64-unknown-unknown-sycldevice -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  return 0;
}

// CHECK: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE10000_clES2_EUlvE10000_
// CHECK: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE10001_clES2_EUlvE10000_
