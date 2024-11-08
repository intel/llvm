// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA scheduler_target_fmax_mhz function attribute.
#include "sycl.hpp"

sycl::queue deviceQueue;

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral '6'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void func2() {}

// Test that checks template parameter support on class member function.
template <int N>
class KernelFunctor2 {
public:
  [[intel::scheduler_target_fmax_mhz(N)]] void operator()() const {
  }
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
    h.single_task<class kernel_name_3>(
        []() [[intel::scheduler_target_fmax_mhz(4)]]{});

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_5>(
        // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_5
        // CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
        // CHECK-NEXT: ConstantExpr {{.*}} 'int'
        // CHECK-NEXT: value: Int 6
        // CHECK-NEXT: IntegerLiteral{{.*}}6{{$}}
        []() [[intel::scheduler_target_fmax_mhz(6),
               intel::scheduler_target_fmax_mhz(6)]]{});
  });

  func2<6>();

  return 0;
}
