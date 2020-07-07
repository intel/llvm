// RUN: %clang %s -fsyntax-only -Xclang -ast-dump -fsycl-device-only | FileCheck %s

[[intelfpga::no_global_work_offset]] // expected-no-diagnostics
void
func();

[[intelfpga::max_work_group_size(8, 8, 8)]] // expected-no-diagnostics
void
func();

[[cl::reqd_work_group_size(4, 4, 4)]] // expected-no-diagnostics
void
func() {}

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(Type bar) {
  bar();
  func();
}

template <typename Name, typename Type>
void parallel_for(Type func) {
  __my_kernel__<Name>(func);
}

void invoke_foo2() {
  // CHECK-LABEL:  FunctionDecl {{.*}} invoke_foo2 'void ()'
  // CHECK:  `-FunctionDecl {{.*}} _ZTSZ11invoke_foo2vE10KernelName 'void ()'
  // CHECK:  -SYCLIntelMaxWorkGroupSizeAttr {{.*}} Inherited 8 8 8
  // CHECK:  -SYCLIntelNoGlobalWorkOffsetAttr {{.*}} Inherited Enabled
  // CHECK:  `-ReqdWorkGroupSizeAttr {{.*}} 4 4 4
  parallel_for<class KernelName>([]() {});
}
