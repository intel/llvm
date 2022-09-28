// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

void myFunc() { }

int main() {

  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function()

  // CHECK: call spir_func void @_Z6myFuncv()

  // CHECK: define {{.*}}spir_func void @_Z6myFuncv()

  kernel_single_task<class kernel_function>([]() { myFunc(); });
  return 0;
}
