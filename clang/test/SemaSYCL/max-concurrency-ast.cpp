// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA max concurrency function attribute.
#include "sycl.hpp"

using namespace sycl;
queue q;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
[[intel::max_concurrency(1)]] void func1() {}

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
[[intel::max_concurrency(0)]] void func2() {}

// CHECK: FunctionTemplateDecl {{.*}} func3
// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: TemplateArgument integral '5'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
template <int N>
[[intel::max_concurrency(N)]] void func3() {}

class KernelFunctor {
public:
  void operator()() const {
    func1();
    func2();
  }
};

template <int N>
class KernelFunctor2 {
public:
  [[intel::max_concurrency(N)]] void operator()() const {
  }
};

void foo() {
  q.submit([&](handler &h) {
    // Test attribute is not propagated.
    // CHECK: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLIntelMaxConcurrencyAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelMaxConcurrencyAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);

    // CHECK: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelMaxConcurrencyAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
    h.single_task<class kernel_name_3>(
        []() [[intel::max_concurrency(4)]] {});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_4>(
        // CHECK: FunctionDecl {{.*}}kernel_name_4
        // CHECK: SYCLIntelMaxConcurrencyAttr
        // CHECK-NEXT: ConstantExpr {{.*}} 'int'
        // CHECK-NEXT: value: Int 3
        // CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
        // CHECK-NOT: SYCLIntelMaxConcurrencyAttr
        []() [[intel::max_concurrency(3),
               intel::max_concurrency(3)]] {});
  });

  func3<5>();
}
