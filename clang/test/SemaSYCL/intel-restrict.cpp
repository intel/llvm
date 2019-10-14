// RUN: %clang %s -fsyntax-only --sycl -DCHECKDIAG -Xclang -verify
// RUN: %clang %s -fsyntax-only -Xclang -ast-dump --sycl | FileCheck %s

[[intel::kernel_args_restrict]] // expected-warning{{'kernel_args_restrict' attribute ignored}}
void func_ignore() {}


struct Functor {
  [[intel::kernel_args_restrict]]
  void operator()() {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel1
  // CHECK:       SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel1>(
      Functor());

  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel2
  // CHECK:       SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel2>(
      []() [[intel::kernel_args_restrict]] {});

  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel3
  // CHECK-NOT:   SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel3>(
      []() {func_ignore();});
}
