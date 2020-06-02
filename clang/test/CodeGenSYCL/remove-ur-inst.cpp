// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

SYCL_EXTERNAL void doesNotReturn() throw() __attribute__((__noreturn__));

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel<class test>([]() {
    doesNotReturn();
    // CHECK-NOT: unreachable
  });
  return 0;
}