// RUN: %clang_cc1 %s -fsyntax-only -fsycl -fsycl-is-device -triple spir64 -verify
// RUN: %clang_cc1 %s -fsyntax-only -fsycl -fsycl-is-device -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 | FileCheck %s

#ifndef TRIGGER_ERROR
//first case - good case
[[intelfpga::no_global_work_offset]] // expected-no-diagnostics
void
func1();

[[intelfpga::max_work_group_size(4, 4, 4)]] void func1();

[[cl::reqd_work_group_size(2, 2, 2)]] void func1() {}

#else
//second case - expect error
[[intelfpga::max_work_group_size(4, 4, 4)]] void
func2();

[[cl::reqd_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
func2() {}

//third case - expect error
[[cl::reqd_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func3();

[[cl::reqd_work_group_size(1, 1, 1)]] // expected-note 2 {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'reqd_work_group_size' is already applied with different parameters}}
func3() {} // expected-error {{'reqd_work_group_size' attribute conflicts with ''reqd_work_group_size'' attribute}}

//fourth case - expect error
[[intelfpga::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func4();

[[intelfpga::max_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'max_work_group_size' is already applied with different parameters}}
func4(); // expected-error {{'max_work_group_size' attribute conflicts with ''max_work_group_size'' attribute}}
#endif

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(Type bar) {
  bar();
#ifndef TRIGGER_ERROR
  func1();
#else
  func2();
  func3();
#endif
}

template <typename Name, typename Type>
void parallel_for(Type func) {
  __my_kernel__<Name>(func);
}

void invoke_foo2() {
#ifndef TRIGGER_ERROR
  // CHECK-LABEL:  FunctionDecl {{.*}} invoke_foo2 'void ()'
  // CHECK:  `-FunctionDecl {{.*}} _ZTSZ11invoke_foo2vE10KernelName 'void ()'
  // CHECK:  -SYCLIntelMaxWorkGroupSizeAttr {{.*}} Inherited 4 4 4
  // CHECK:  -SYCLIntelNoGlobalWorkOffsetAttr {{.*}} Inherited Enabled
  // CHECK:  `-ReqdWorkGroupSizeAttr {{.*}} 2 2 2
  parallel_for<class KernelName>([]() {});
#else
  parallel_for<class KernelName>([]() {}); // expected-error {{conflicting attributes applied to a SYCL kernel or SYCL_EXTERNAL function}}
#endif
}
