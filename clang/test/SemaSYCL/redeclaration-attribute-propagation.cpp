// RUN: %clang_cc1 %s -fsyntax-only -fsycl -fsycl-is-device -triple spir64 -Wno-sycl-2017-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -fsycl -fsycl-is-device -triple spir64 -DTRIGGER_ERROR -Wno-sycl-2017-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 -Wno-sycl-2017-compat | FileCheck %s

#include "Inputs/sycl.hpp"

#ifndef TRIGGER_ERROR
//first case - good case
[[INTEL::no_global_work_offset]] // expected-no-diagnostics
void
func1();

[[INTEL::max_work_group_size(4, 4, 4)]] void func1();

[[cl::reqd_work_group_size(2, 2, 2)]] void func1() {}

#else
//second case - expect error
[[INTEL::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func2();

[[cl::reqd_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
func2() {}

//third case - expect error
[[cl::reqd_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func3();

[[cl::reqd_work_group_size(1, 1, 1)]] // expected-note {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'reqd_work_group_size' is already applied with different parameters}}
func3() {} // expected-error {{'reqd_work_group_size' attribute conflicts with ''reqd_work_group_size'' attribute}}

//fourth case - expect error
[[INTEL::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func4();

[[INTEL::max_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'max_work_group_size' is already applied with different parameters}}
func4() {} // expected-error {{'max_work_group_size' attribute conflicts with ''max_work_group_size'' attribute}}
#endif

int main() {
#ifndef TRIGGER_ERROR
  // CHECK-LABEL:  FunctionDecl {{.*}} main 'int ()'
  // CHECK:  `-FunctionDecl {{.*}}test_kernel1 'void ()'
  // CHECK:  -SYCLIntelMaxWorkGroupSizeAttr {{.*}} Inherited 4 4 4
  // CHECK:  -SYCLIntelNoGlobalWorkOffsetAttr {{.*}} Inherited Enabled
  // CHECK:  `-ReqdWorkGroupSizeAttr {{.*}} 2 2 2
  cl::sycl::kernel_single_task<class test_kernel1>(
      []() { func1(); });

#else
  cl::sycl::kernel_single_task<class test_kernel2>(
      []() { func2(); }); // expected-error {{conflicting attributes applied to a SYCL kernel or SYCL_EXTERNAL function}}

  cl::sycl::kernel_single_task<class test_kernel3>(
      []() { func3(); });

  cl::sycl::kernel_single_task<class test_kernel4>(
      []() { func4(); });
#endif
}
