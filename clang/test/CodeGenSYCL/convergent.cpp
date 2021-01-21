// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -disable-llvm-passes \
// RUN:  -triple spir64 -emit-llvm %s -o - | FileCheck %s

// CHECK-DAG: Function Attrs:
// CHECK-DAG-SAME: convergent
// CHECK-DAG-NEXT: define void @_Z3foov
void foo() {
  int a = 1;
}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { foo(); });
  return 0;
}
