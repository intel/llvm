// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {});
  return 0;
}

// CHECK: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CHECK: !spirv.Source = !{[[LANG:![0-9]+]]}
// CHECK: [[SPIR]] = !{i32 1, i32 2}
// CHECK: [[LANG]] = !{i32 4, i32 100000}
