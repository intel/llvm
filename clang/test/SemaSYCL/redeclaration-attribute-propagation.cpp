// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -Wno-sycl-2017-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -DTRIGGER_ERROR -Wno-sycl-2017-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -Wno-sycl-2017-compat | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef TRIGGER_ERROR
//first case - good case
[[intel::no_global_work_offset]] // expected-no-diagnostics
void
func1();

[[intel::max_work_group_size(4, 4, 4)]] void func1();

[[sycl::reqd_work_group_size(2, 2, 2)]] void func1() {}

#else
//second case - expect error
[[intel::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func2();

[[sycl::reqd_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
func2() {}

//third case - expect error
[[sycl::reqd_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func3();

[[sycl::reqd_work_group_size(1, 1, 1)]] // expected-note {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
func3() {} // expected-error {{'reqd_work_group_size' attribute conflicts with ''reqd_work_group_size'' attribute}}

//fourth case - expect error
[[intel::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
void
func4();

[[intel::max_work_group_size(8, 8, 8)]] // expected-note {{conflicting attribute is here}}
void
// expected-warning@+1 {{attribute 'max_work_group_size' is already applied with different arguments}}
func4() {} // expected-error {{'max_work_group_size' attribute conflicts with ''max_work_group_size'' attribute}}
#endif

int main() {
  q.submit([&](handler &h) {
#ifndef TRIGGER_ERROR
    // CHECK-LABEL:  FunctionDecl {{.*}} main 'int ()'
    // CHECK:  `-FunctionDecl {{.*}}test_kernel1 'void ()'
    // CHECK:  -SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK:  -SYCLIntelMaxWorkGroupSizeAttr {{.*}} Inherited
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK:  `-ReqdWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class test_kernel1>(
        []() { func1(); });

#else
    h.single_task<class test_kernel2>(
        []() { func2(); }); // expected-error {{conflicting attributes applied to a SYCL kernel or SYCL_EXTERNAL function}}

    h.single_task<class test_kernel3>(
        []() { func3(); });

    h.single_task<class test_kernel4>(
        []() { func4(); });
#endif
  });
  return 0;
}
