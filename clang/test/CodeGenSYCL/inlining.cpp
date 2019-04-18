// RUN: %clang --sycl %s -S -emit-llvm -o - | FileCheck %s

template <typename name, typename Func>
__kernel void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-NOT: noinline
  kernel_single_task<class kernel_function>([]() {});
  return 0;
}
