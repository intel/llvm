// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA fpga_pipeline function attribute.
#include "sycl.hpp"

sycl::queue deviceQueue;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelFpgaPipelineAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
[[intel::fpga_pipeline(0)]] void func1() {}

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK_NEXT: SYCLIntelFpgaPipelineAttr
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral 1
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelFpgaPipelineAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
template <int N>
[[intel::fpga_pipeline(N)]] void func2() {}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelFpgaPipelineAttr
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
[[intel::fpga_pipeline]]
[[intel::fpga_pipeline]] void func3() {}

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
  [[intel::fpga_pipeline(N)]] void operator()() const {
  }
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLIntelFpgaPipelineAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelFpgaPipelineAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    KernelFunctor2<1> f2;
    h.single_task<class kernel_name_2>(f2);
    
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelFpgaPipelineAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    h.single_task<class kernel_name_3>(
        []() [[intel::fpga_pipeline]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_4>(
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_4
    // CHECK: SYCLIntelFpgaPipelineAttr
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 0
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
        []() [[intel::fpga_pipeline(0),
               intel::fpga_pipeline(0)]]{});
  });

  func2<1>();

  return 0;
}
