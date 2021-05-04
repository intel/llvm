// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DCHECKDIAG -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -Wno-sycl-2017-compat -sycl-std=2017 -triple spir64 -DSYCL2017 %s
// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DCHECKDIAG -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -Wno-sycl-2017-compat -sycl-std=2020 -triple spir64 -DSYCL2020 %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only -Wno-sycl-strict -DNODIAG %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only -sycl-std=2020 -Wno-sycl-strict -DNODIAG %s

[[intel::kernel_args_restrict]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::kernel_args_restrict]] void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
#ifdef CHECKDIAG
  [[intel::kernel_args_restrict]] int invalid = 42; // expected-error{{'kernel_args_restrict' attribute only applies to functions}}
#endif
}

int main() {
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
  // CHECK:       SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel1>(
      FuncObj());

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
  // CHECK:       SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel2>(
      []() [[intel::kernel_args_restrict]] {});

#if defined(SYCL2017)
  // Test attribute is propagated.
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
  // CHECK:       SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel3>(
      []() { func_do_not_ignore(); });
#endif // SYCL2017

#if defined(SYCL2020)
  // Test attribute is not propagated.
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
  // CHECK-NOT:   SYCLIntelKernelArgsRestrictAttr
  kernel<class test_kernel4>(
      []() { func_do_not_ignore(); });
#endif // SYCL2020
}
#if defined(NODIAG)
// expected-no-diagnostics
#endif
