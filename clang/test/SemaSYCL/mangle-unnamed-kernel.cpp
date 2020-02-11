// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -I %S/Inputs -fsycl-is-device -fsycl-unnamed-lambda -ast-dump %s | FileCheck %s
#include <sycl.hpp>

int main() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  return 0;
}

// CHECK: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE6->12clES2_EUlvE6->54{{.*}}
// CHECK: _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE7->12clES2_EUlvE7->54{{.*}}
