// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-unknown-windows-unknown -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck --check-prefix CHK-WIN %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-unknown-linux-unknown -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck --check-prefix CHK-LIN %s

#include "Inputs/sycl.hpp"
// CHK-WIN: %struct{{.*}}F = type { i8, i8 }
// CHK-LIN: %struct{{.*}}F = type { i8 }
struct F1 {};
struct F2 {};
struct F : F1, F2 {
  char x;
};

int main() {
  cl::sycl::accessor<F, 1, cl::sycl::access::mode::read_write> accessorA;
  cl::sycl::handler cgh;
  cgh.single_task<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
