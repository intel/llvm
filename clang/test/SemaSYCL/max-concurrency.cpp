// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020  -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;

class Functor1 {
public:
  [[intel::max_concurrency(4)]] void operator()() const {}
};

[[intel::max_concurrency]] void foo() {} // expected-error {{'max_concurrency' attribute takes one argument}}

class Functor2 {
public:
  void operator()() const {
    foo();
  }
};

template <int NT>
class Functor3 {
public:
  [[intel::max_concurrency(NT)]] void operator()() const {}
  // expected-error@+1 {{'max_concurrency' attribute only applies to 'for', 'while', 'do' statements, and functions}}
  [[intel::max_concurrency(2)]] int a[10];
};

// expected-error@+1 {{'max_concurrency' attribute takes one argument}}
[[intel::max_concurrency(3, 3)]] void goo() {}

class Functor4 {
public:
  void operator() () const {
    goo();
  }
};

// expected-error@+1 {{'max_concurrency' attribute requires a positive integral compile time constant expression}}
[[intel::max_concurrency(-1)]] void bar() {}
class Functor5 {
public:
  void operator() () const {
    bar();
  }
};

// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'const char [16]'}}
[[intel::max_concurrency("numberofthreads")]] void zoo() {}

template <int NT>
[[intel::max_concurrency(NT)]] void func() {}

[[intel::max_concurrency(8)]] void dup();   // expected-note {{previous attribute is here}}
[[intel::max_concurrency(9)]] void dup() {} // expected-warning {{attribute 'max_concurrency' is already applied with different arguments}}

int main() {
  queue q;

  q.submit([&](handler &h) {
    Functor1 f1;
    h.single_task<class kernel_name1>(f1);

    Functor2 f2;
    h.single_task<class kernel_name2>(f2);

    // Applying attributes to lambdas is a nonconforming extension. This will
    // remain an an error until the SYCL specifications allow it or until
    // users require it. Refer to comments about
    // SupportsNonconformingLambdaSyntax bit in Attr.td.
    h.single_task<class kernel_name3>(
      []() [[intel::max_concurrency(3)]]{}); // expected-error{{'max_concurrency' attribute cannot be applied to types}}

    Functor3<4> f3;
    h.single_task<class kernel_name4>(f3);

    h.single_task<class kernel_name5>([]() {
      func<5>();
    });

  });
}

// CHECK: CXXMethodDecl {{.*}}used operator() {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 4
// CHECK: IntegerLiteral {{.*}}4{{$}}
// CHECK: CXXMethodDecl {{.*}}operator() {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'NT' 'int'
// CHECK: CXXMethodDecl {{.*}}{{.*}}used operator() {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 4
// CHECK: IntegerLiteral {{.*}}4{{$}}
// CHECK: FunctionDecl {{.*}}{{.*}}func {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: FunctionDecl {{.*}}{{.*}}used func 'void ()'
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 5
// CHECK: IntegerLiteral {{.*}}5{{$}}
// CHECK: FunctionDecl {{.*}}{{.*}}dup {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 8
// CHECK: IntegerLiteral {{.*}}8{{$}}
// CHECK: FunctionDecl {{.*}}{{.*}}dup {{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 9
// CHECK: IntegerLiteral {{.*}}9{{$}}
// CHECK: FunctionDecl {{.*}}{{.*}}kernel_name1{{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 4
// CHECK: IntegerLiteral{{.*}}4{{$}}
// CHECK: FunctionDecl {{.*}}{{.*}}kernel_name4{{.*}}
// CHECK: SYCLIntelFPGAMaxConcurrencyAttr {{.*}}
// CHECK: ConstantExpr {{.*}} 'int'
// CHECK: value: Int 4
// CHECK: IntegerLiteral{{.*}}4{{$}}
