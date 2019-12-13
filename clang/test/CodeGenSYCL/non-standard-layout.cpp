// RUN: %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -aux-triple x86_64-unknown-windows-unknown -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck --check-prefix CHK-WIN %s
// RUN: %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -aux-triple x86_64-unknown-linux-unknown -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck --check-prefix CHK-LIN %s

#include "sycl.hpp"
// CHK-WIN: %struct{{.*}}F = type { i8, i8 }
// CHK-LIN: %struct{{.*}}F = type { i8 }
struct F1 {};
struct F2 {};
struct F : F1, F2 {
  char x;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<F, 1, cl::sycl::access::mode::read_write> accessorA;
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
