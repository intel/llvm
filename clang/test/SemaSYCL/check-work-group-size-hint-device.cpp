// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify -DEXPECT_PROP -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -DEXPECT_PROP %s | FileCheck %s

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Test for AST of work_group_size_hint kernel attribute in SYCL. 
#include "sycl.hpp"

// Check the basics.
#ifdef TRIGGER_ERROR
[[sycl::work_group_size_hint]] void f0();                 // expected-error {{'work_group_size_hint' attribute takes at least 1 argument}}
[[sycl::work_group_size_hint(12, 12, 12, 12)]] void f1(); // expected-error {{'work_group_size_hint' attribute takes no more than 3 arguments}}
[[sycl::work_group_size_hint("derp", 1, 2)]] void f2();   // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'const char[5]'}}
[[sycl::work_group_size_hint(1, 1, 1)]] int i;            // expected-error {{'work_group_size_hint' attribute only applies to functions}}
#endif

// Produce a conflicting attribute warning when the args are different.
[[sycl::work_group_size_hint(4, 1, 1)]] void f3();    // expected-note {{previous attribute is here}}
[[sycl::work_group_size_hint(1, 1, 32)]] void f3() {} // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

// 1 and 2 dim versions
[[sycl::work_group_size_hint(2)]] void f4();    // ok
[[sycl::work_group_size_hint(2, 1)]] void f5(); // ok

// Redefinitions with different dimensionality.
[[sycl::work_group_size_hint(2, 1)]] void f6();    // expected-note {{previous attribute is here}}
[[sycl::work_group_size_hint(2)]] void f6();       // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}} expected-note {{previous attribute is here}}
[[sycl::work_group_size_hint(2, 1, 1)]] void f6(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
[[sycl::work_group_size_hint(4, 1, 1), sycl::work_group_size_hint(32, 1, 1)]] void f7(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}} expected-note {{previous attribute is here}}

// Show that the attribute works on member functions.
class Functor_1 {
public:
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(16, 1, 1)]] void operator()() const;
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(32, 1, 1)]] void operator()(int) const; // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}} expected-note {{previous attribute is here}}
};

// Ensure that template arguments behave appropriately based on instantiations.
template <int N>
[[sycl::work_group_size_hint(N, 1, 1)]] void f8(); // #f8

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(1, 1, 1)]] void f9(); // #f9prev

template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, Y, Z)]] void f9() {} // #f9

// Test that a template redeclaration where the difference is known up front is
// diagnosed immediately, even without instantiation.
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, 1, Z)]] void f10(); // expected-note {{previous attribute is here}}
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, 2, Z)]] void f10(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

#ifdef TRIGGER_ERROR
[[sycl::work_group_size_hint(1, 2, 0)]] void f11(); // expected-error {{'work_group_size_hint' attribute requires a positive integral compile time constant expression}}
#endif

void instantiate() {
  f8<1>(); // OK
#ifdef TRIGGER_ERROR
  // expected-error@#f8 {{'work_group_size_hint' attribute requires a positive integral compile time constant expression}}
  f8<-1>(); // expected-note {{in instantiation}}
  // expected-error@#f8 {{'work_group_size_hint' attribute requires a positive integral compile time constant expression}}
  f8<0>(); // expected-note {{in instantiation}}
#endif

  f9<1, 1, 1>(); // OK, args are the same on the redecl.

  // expected-warning@#f9 {{attribute 'work_group_size_hint' is already applied with different arguments}}
  // expected-note@#f9prev {{previous attribute is here}}
  f9<1, 2, 3>(); // expected-note {{in instantiation}}
}

// Show that the attribute works on member functions.
class Functor16x2x1 {
public:
  [[sycl::work_group_size_hint(16, 2, 1)]] void operator()() const {};
};

// CHECK: CXXRecordDecl {{.*}} {{.*}}Functor16x2x1
// CHECK: SYCLWorkGroupSizeHintAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 2
// CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

class Functor4x4x4 {
public:
  [[sycl::work_group_size_hint(4, 4, 4)]] void operator()() const {};
};

// Checking whether propagation of the attribute happens or not, according to the SYCL version.
#if defined(EXPECT_PROP) // if attribute is propagated, then we expect errors here
void f8x8x8(){};
#else // otherwise no error
[[sycl::work_group_size_hint(8, 8, 8)]] void f8x8x8(){};
#endif
class FunctorNoProp {
public:
  void operator()() const {
    f8x8x8();
  };
};

void invoke() {
  Functor16x2x1 f16x2x1;
  Functor4x4x4 f4x4x4;

  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel_1>(f16x2x1);
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_1
    // CHECK: SYCLWorkGroupSizeHintAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}


    FunctorNoProp fNoProp;
    h.single_task<class kernel_3>(fNoProp);
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_3
    // CHECK-NOT: SYCLWorkGroupSizeHintAttr

    h.single_task<class kernel_name4>([]() [[sycl::work_group_size_hint(4,4,4)]] {});
    // CHECK: FunctionDecl {{.*}}kernel_name4
    // CHECK: SYCLWorkGroupSizeHintAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

  });

  // FIXME: Add tests with the C++23 lambda attribute syntax.
}
