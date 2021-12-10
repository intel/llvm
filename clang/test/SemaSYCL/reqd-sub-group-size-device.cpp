// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::reqd_sub_group_size(4)]] void foo() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(32)]] void baz() {} // expected-note {{conflicting attribute is here}}

// No diagnostic is emitted because the arguments match.
[[intel::reqd_sub_group_size(12)]] void bar();
[[intel::reqd_sub_group_size(12)]] void bar() {} // OK

// No diagnostic because the attributes are synonyms with identical behavior.
[[sycl::reqd_sub_group_size(12)]] void bar(); // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::reqd_sub_group_size(12)]] void quux(); // expected-note {{previous attribute is here}}
[[intel::reqd_sub_group_size(100)]] void quux(); // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}} expected-note {{previous attribute is here}}
[[sycl::reqd_sub_group_size(200)]] void quux(); // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}}

#ifdef TRIGGER_ERROR
// Make sure there's at least one argument passed.
[[sycl::reqd_sub_group_size]] void quibble(); // expected-error {{'reqd_sub_group_size' attribute takes one argument}}
#endif // TRIGGER_ERROR

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[intel::reqd_sub_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    foo();
  }
};

class Functor4 {
public:
  [[intel::reqd_sub_group_size(12)]] void operator()() const {}
};

class Functor {
public:
  void operator()() const {
    foo();
  }
};

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor f;
    h.single_task<class kernel_name2>(f);

#ifdef TRIGGER_ERROR
    Functor8 f8;
    h.single_task<class kernel_name3>(f8);

    h.single_task<class kernel_name4>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
      foo();
      baz();
    });
#endif

    h.single_task<class kernel_name5>([]() [[intel::reqd_sub_group_size(2)]]{});
    h.single_task<class kernel_name6>([]() [[intel::reqd_sub_group_size(4)]] { foo(); });
    h.single_task<class kernel_name7>([]() [[intel::reqd_sub_group_size(6)]]{});

    Functor4 f4;
    h.single_task<class kernel_name8>(f4);
  });
  return 0;
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B();
[[intel::reqd_sub_group_size(16)]] void A() {
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B() {
  A();
}

#ifdef TRIGGER_ERROR
// expected-note@+1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(2)]] void sg_size2() {}

// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{conflicting attributes applied to a SYCL kernel}}
[[intel::reqd_sub_group_size(4)]] __attribute__((sycl_device)) void sg_size4() {
  sg_size2();
}
#endif

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 16
// CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 4
// CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name7
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: IntegerLiteral{{.*}}6{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name8
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 12
// CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
