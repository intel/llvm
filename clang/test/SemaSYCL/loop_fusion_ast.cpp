// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA loop fusion function attributes
#include "sycl.hpp"

using namespace cl::sycl;
queue q;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
[[intel::loop_fuse]] void func1() {}

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
[[intel::loop_fuse(0)]] void func2() {}

// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
[[intel::loop_fuse_independent]] void func3() {}

// CHECK: FunctionDecl {{.*}} func4 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
[[intel::loop_fuse_independent(3)]] void func4() {}

// CHECK: FunctionTemplateDecl {{.*}} func5
// CHECK: FunctionDecl {{.*}} func5 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK_NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func5 'void ()'
// CHECK-NEXT: TemplateArgument integral 1
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
template <int N>
[[intel::loop_fuse(N)]] void func5() {}

// CHECK: FunctionTemplateDecl {{.*}} func6
// CHECK: FunctionDecl {{.*}} func6 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK_NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func6 'void ()'
// CHECK-NEXT: TemplateArgument integral 5
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
template <int N>
[[intel::loop_fuse_independent(N)]] void func6() {}

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
    // CHECK: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLIntelLoopFuseAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);

    // CHECK: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    h.single_task<class kernel_name_3>(
        []() [[intel::loop_fuse_independent]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_4>(
    // CHECK: FunctionDecl {{.*}}kernel_name_4
    // CHECK: SYCLIntelLoopFuseAttr {{.*}} loop_fuse
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
        []() [[intel::loop_fuse(3),
               intel::loop_fuse(3)]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_5>(
    // CHECK: FunctionDecl {{.*}}kernel_name_5
    // CHECK: SYCLIntelLoopFuseAttr {{.*}} loop_fuse_independent
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
        []() [[intel::loop_fuse_independent,
               intel::loop_fuse_independent]]{});
  });

  func5<1>();
  func6<5>();
}
