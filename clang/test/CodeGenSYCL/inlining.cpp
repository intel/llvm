// RUN: %clang_cc1 -O1 -fsycl-is-device -triple spir64-unknown-unknown %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -fsycl-is-device -triple spir64-unknown-unknown %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-O0

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-O0: noinline
  // CHECK-NOT: noinline
  kernel_single_task<class kernel_function>([]() {});
  return 0;
}
