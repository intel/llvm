// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-pc-windows-msvc -I %S/../Headers/Inputs/include/ -ast-dump %s | FileCheck %s --check-prefix=CHECK-64-WIN
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-unknown-linux-gnu -I %S/../Headers/Inputs/include/ -ast-dump %s | FileCheck %s --check-prefix=CHECK-64-LIN
// RUN: %clang_cc1 -fsycl-is-device -triple spir-unknown-linux-sycldevice -I %S/../Headers/Inputs/include/ -ast-dump %s | FileCheck %s --check-prefix=CHECK-32
#include "Inputs/sycl.hpp"
#include <stdlib.h>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

template <typename T>
class SimpleVadd;

int main() {
  kernel<class SimpleVadd<int>>(
      [=](){});

  kernel<class SimpleVadd<double>>(
      [=](){});

  kernel<class SimpleVadd<size_t>>(
      [=](){});
  return 0;
}

// CHECK: _ZTS10SimpleVaddIiE
// CHECK: _ZTS10SimpleVaddIdE
// CHECK-64-WIN: _ZTS10SimpleVaddIyE
// CHECK-64-LIN: _ZTS10SimpleVaddImE
// CHECK-32: _ZTS10SimpleVaddIjE
