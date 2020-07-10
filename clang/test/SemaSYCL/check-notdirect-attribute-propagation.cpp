// RUN: %clang %s -fsyntax-only -Xclang -ast-dump -fsycl-device-only | FileCheck %s

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
  // CHECK:  `-FunctionDecl {{.*}} _ZTSZ11invoke_foo2vE10KernelName 'void ()'
  // CHECK:  -SYCLIntelNoGlobalWorkOffsetAttr {{.*}} Enabled
  parallel_for<class KernelName>([]() {});
}
