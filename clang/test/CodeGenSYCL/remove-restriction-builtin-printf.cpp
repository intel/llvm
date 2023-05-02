// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s
// This test checks if __builtin_printf is emitted in the IR.

#include "sycl.hpp"

using namespace sycl;
queue q;

int main() {
  q.submit([&](handler &h) {
    // CHECK: define {{.*}}spir_kernel {{.*}}
    h.single_task<class kernelA>([=]() {
      // CHECK: call {{.*}}printf(ptr noundef @{{.*}}, i32 noundef 24)
        __builtin_printf("hello, %d\n", 24);
    });
  });
  return 0;
}
