// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

// Test for AST of reqd_work_group_size kernel attribute in SYCL 1.2.1.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

#ifndef __SYCL_DEVICE_ONLY__
// expected-no-diagnostics
class Functor {
public:
  [[sycl::reqd_work_group_size(4)]] void operator()() const {}
};

void bar() {
  q.submit([&](handler &h) {
    Functor f;
    h.single_task<class kernel_name>(f);
  });
}

#else
[[sycl::reqd_work_group_size(4)]] void f4x1x1() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32)]] void f32x1x1() {} // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16)]] void f16x1x1() {}      // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16, 16)]] void f16x16x1() {} // expected-note {{conflicting attribute is here}}

[[sycl::reqd_work_group_size(32, 32)]] void f32x32x1() {}      // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}

class Functor33 {
public:
  // expected-warning@+1{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
  [[sycl::reqd_work_group_size(32, -4)]] void operator()() const {}
};

class Functor30 {
public:
  // expected-warning@+1 2{{implicit conversion changes signedness: 'int' to 'unsigned long long'}}
  [[sycl::reqd_work_group_size(30, -30, -30)]] void operator()() const {}
};

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const {}
};

class Functor64 {
public:
  [[sycl::reqd_work_group_size(64, 64)]] void operator()() const {}
};

class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[sycl::reqd_work_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    f4x1x1();
  }
};

class Functor {
public:
  void operator()() const {
    f4x1x1();
  }
};

class FunctorAttr {
public:
  [[sycl::reqd_work_group_size(128, 128, 128)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor f;
    h.single_task<class kernel_name2>(f);

    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name3>(f16x16x16);

    FunctorAttr fattr;
    h.single_task<class kernel_name4>(fattr);

    Functor33 f33;
    h.single_task<class kernel_name5>(f33);

    Functor30 f30;
    h.single_task<class kernel_name6>(f30);

    h.single_task<class kernel_name7>([]() [[sycl::reqd_work_group_size(32, 32, 32)]] {
      f32x32x32();
    });
#ifdef TRIGGER_ERROR
    Functor8 f8;
    h.single_task<class kernel_name8>(f8);

    h.single_task<class kernel_name9>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f4x1x1();
      f32x1x1();
    });

    h.single_task<class kernel_name10>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f16x1x1();
      f16x16x1();
    });

    h.single_task<class kernel_name11>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f32x32x32();
      f32x32x1();
    });

    // expected-error@+1 {{expected variable name or 'this' in lambda capture list}}
    h.single_task<class kernel_name12>([[sycl::reqd_work_group_size(32, 32, 32)]][]() {
      f32x32x32();
    });

#endif // TRIGGER_ERROR
  });
  return 0;
}

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 4
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 128
// CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int -4
// CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name6
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 30
// CHECK-NEXT:  IntegerLiteral{{.*}}30{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int -30
// CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
// CHECK-NEXT:  IntegerLiteral{{.*}}30{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int -30
// CHECK-NEXT:  UnaryOperator{{.*}} 'int' prefix '-'
// CHECK-NEXT:  IntegerLiteral{{.*}}30{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name7
// CHECK: ReqdWorkGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 32
// CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
#endif // __SYCL_DEVICE_ONLY__
