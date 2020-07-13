// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 | FileCheck %s

[[intelfpga::no_global_work_offset]] void not_direct() {}

void func() { not_direct(); }

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(Type bar) {
  bar();
  func();
}

template <typename Name, typename Type>
void parallel_for(Type lambda) {
  __my_kernel__<Name>(lambda);
}

void invoke_foo2() {
  // CHECK-LABEL:  FunctionDecl {{.*}} invoke_foo2 'void ()'
  // CHECK:        `-FunctionDecl {{.*}}KernelName 'void ()'
  // CHECK:        -SYCLIntelNoGlobalWorkOffsetAttr {{.*}} Enabled
  parallel_for<class KernelName>([]() {});
}
