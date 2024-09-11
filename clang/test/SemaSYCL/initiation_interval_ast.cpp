// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA initiation_interval function attributes.
#include "sycl.hpp"

sycl::queue deviceQueue;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 4
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
[[intel::initiation_interval(4)]] void func1() {}

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral '6'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
template <int N>
[[intel::initiation_interval(N)]] void func2() {}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 10
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}
[[intel::initiation_interval(10)]]
[[intel::initiation_interval(10)]] void func3() {}

class KernelFunctor {
public:
  void operator()() const {
    func1();
  }
};

// Test that checks template parameter support on class member function.
template <int N>
class KernelFunctor2 {
public:
  [[intel::initiation_interval(N)]] void operator()() const {
  }
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLIntelInitiationIntervalAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);
    
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
    h.single_task<class kernel_name_3>(
        []() [[intel::initiation_interval(4)]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_4>(
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_4
    // CHECK: SYCLIntelInitiationIntervalAttr {{.*}} initiation_interval
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 6
    // CHECK-NEXT: IntegerLiteral{{.*}}6{{$}}
        []() [[intel::initiation_interval(6),
               intel::initiation_interval(6)]]{});
  });

  func2<6>();

  return 0;
}
