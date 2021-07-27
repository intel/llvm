// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

// Test for AST of reqd_work_group_size kernel attribute in SYCL 1.2.1.
#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[sycl::reqd_work_group_size(4, 1, 1)]] void f4x1x1() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32, 1, 1)]] void f32x1x1() {} // expected-note {{conflicting attribute is here}}

[[sycl::reqd_work_group_size(16, 1, 1)]] void f16x1x1() {} // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16, 16, 1)]] void f16x16x1() {} // expected-note {{conflicting attribute is here}}

[[sycl::reqd_work_group_size(32, 32, 1)]] void f32x32x1() {} // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {} // expected-note {{conflicting attribute is here}}

// No diagnostic because the attributes are synonyms with identical behavior.
[[sycl::reqd_work_group_size(4, 4, 4)]] void four();
[[sycl::reqd_work_group_size(4, 4, 4)]] void four(); // OK

// Same for the default values.
// FIXME: This turns out to be wrong as there aren't really default values
// (that is an implementation detail we use but shouldn't expose to the user).
// Instead, the dimensionality of the attribute needs to match that of the
// kernel, so the one, two, and three arg forms of the attribute are actually
// *different* attributes. This means that you should not be able to redeclare
// the function with a different dimensionality.
[[sycl::reqd_work_group_size(4)]] void four_again();
[[sycl::reqd_work_group_size(4)]] void four_again(); // OK
[[sycl::reqd_work_group_size(4, 1)]] void four_again(); // OK
[[sycl::reqd_work_group_size(4, 1)]] void four_again(); // OK
[[sycl::reqd_work_group_size(4, 1, 1)]] void four_again(); // OK
[[sycl::reqd_work_group_size(4, 1, 1)]] void four_again(); // OK

// Make sure there's at least one argument passed for the SYCL spelling.
#ifdef TRIGGER_ERROR
[[sycl::reqd_work_group_size]] void four_no_more(); // expected-error {{'reqd_work_group_size' attribute takes at least 1 argument}}
#endif // TRIGGER_ERROR

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16, 1, 1)]] [[sycl::reqd_work_group_size(16, 1, 1)]] void operator()() const {}
};

#ifdef TRIGGER_ERROR
class Functor32 {
public:
  // expected-note@+3{{conflicting attribute is here}}
  // expected-warning@+2{{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-error@+1{{'reqd_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(32, 1, 1)]] [[sycl::reqd_work_group_size(1, 1, 32)]] void operator()() const {}
};
#endif
class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[sycl::reqd_work_group_size(1, 1, 8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    f4x1x1();
  }
};

class Functor {
public:
  void operator()() const {
    f4x1x1();
  }
};

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor f;
    h.single_task<class kernel_name2>(f);

    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name3>(f16x16x16);

    h.single_task<class kernel_name5>([]() [[sycl::reqd_work_group_size(32, 32, 32), sycl::reqd_work_group_size(32, 32, 32)]] {
      f32x32x32();
    });

#ifdef TRIGGER_ERROR
    Functor8 f8;
    h.single_task<class kernel_name6>(f8);

    Functor32 f32;
    h.single_task<class kernel_name1>(f32);

    h.single_task<class kernel_name7>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f4x1x1();
      f32x1x1();
    });

    h.single_task<class kernel_name8>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f16x1x1();
      f16x16x1();
    });

    h.single_task<class kernel_name9>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      f32x32x32();
      f32x32x1();
    });

    // expected-error@+1 {{expected variable name or 'this' in lambda capture list}}
    h.single_task<class kernel_name10>([[sycl::reqd_work_group_size(32, 32, 32)]][]() {
      f32x32x32();
    });

#endif
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
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
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
