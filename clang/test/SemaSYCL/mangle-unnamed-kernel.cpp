// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -I %S/Inputs -fsycl-is-device -fsycl-unnamed-lambda -ast-dump %s | FileCheck %s
#include <sycl.hpp>

int main() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });
  return 0;
}

// CHECK: _ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_
// CHECK: _ZTSZZ4mainENK3$_1clERN2cl4sycl7handlerEEUlvE_