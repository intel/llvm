// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s


template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  bool test = false;

  // CHECK: @_ZTSZ4mainE11test_kernel(i8 {{.*}} [[ARG:%[A-Za-z_0-9]*]]
  // CHECK: %__SYCLKernel = alloca
  // CHECK: %test = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %__SYCLKernel.ascast
  // CHECK: store i8 %{{.*}}, ptr addrspace(4) %test
  kernel<class test_kernel>([=]() {
    (void)test;
  });

  return 0;
}
