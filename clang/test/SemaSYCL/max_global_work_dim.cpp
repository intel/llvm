// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA max_global_work_dim function attribute in SYCL 2020 mode.
#include "sycl.hpp"

sycl::queue deviceQueue;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
[[intel::max_global_work_dim(2)]] void func1() {}

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK_NEXT: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral 2
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
template <int N>
[[intel::max_global_work_dim(N)]] void func2() {}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
[[intel::max_global_work_dim(1)]]
[[intel::max_global_work_dim(1)]] void func3() {}

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
  [[intel::max_global_work_dim(N)]] void operator()() const {
  }
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLIntelMaxGlobalWorkDimAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 2
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
    KernelFunctor2<2> f2;
    h.single_task<class kernel_name_2>(f2);
    
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
    h.single_task<class kernel_name_3>(
        []() [[intel::max_global_work_dim(1)]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_4>(
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_4
    // CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 1
    // CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
        []() [[intel::max_global_work_dim(1),
               intel::max_global_work_dim(1)]]{});
  });

  func2<2>();

  return 0;
}
