// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -ast-dump -verify %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr
// CHECK-NEXT: NULL
[[intel::loop_fuse]] void func1() {}

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 0
[[intel::loop_fuse(0)]] void func2() {}

// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseIndependentAttr
// CHECK-NEXT: NULL
[[intel::loop_fuse_independent]] void func3() {}

// CHECK: FunctionDecl {{.*}} func4 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseIndependentAttr
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 3
[[intel::loop_fuse_independent(3)]] void func4() {}

class KernelFunctor {
public:
  void operator()() const {
    func1();
    func3();
  }
};

template <int N>
class KernelFunctor2 {
public:
  [[intel::loop_fuse(N)]] void operator()() const {
  }
};

void foo() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}}kernel_name_1 'void ()'
    // CHECK-NOT: SYCLIntelLoopFuseAttr
    // CHECK-NOT: SYCLIntelLoopFuseIndependentAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK: FunctionDecl {{.*}}kernel_name_2 'void ()'
    // CHECK: SYCLIntelLoopFuseAttr
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);

    // CHECK: FunctionDecl {{.*}}kernel_name_3 'void ()'
    // CHECK: SYCLIntelLoopFuseIndependentAttr
    h.single_task<class kernel_name_3>(
        []() [[intel::loop_fuse_independent]]{});
  });

  [[intel::loop_fuse]] int testVar = 0; // expected-error{{'loop_fuse' attribute only applies to functions}}
}

[[intel::loop_fuse(1048577)]] void func5() {}        // expected-error{{'loop_fuse' attribute requires integer constant between 0 and 1048576 inclusive}}
[[intel::loop_fuse_independent(-1)]] void func6() {} // expected-error{{'loop_fuse_independent' attribute requires integer constant between 0 and 1048576 inclusive}}

[[intel::loop_fuse]] [[intel::loop_fuse(10)]] void func7() {}                     // expected-warning {{attribute 'loop_fuse' is already applied}}
[[intel::loop_fuse_independent]] [[intel::loop_fuse_independent]] void func8() {} // // expected-warning {{attribute 'loop_fuse_independent' is already applied}}

// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse]] [[intel::loop_fuse_independent]] void func9();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] [[intel::loop_fuse]] void func10();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse]] void func11();
[[intel::loop_fuse_independent]] void func11() {}

// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] void func12();
[[intel::loop_fuse]] void func12() {}

[[intel::loop_fuse]] void func13();
[[intel::loop_fuse]] void func13() {}

[[intel::loop_fuse_independent]] void func14();
[[intel::loop_fuse_independent]] void func14() {}
