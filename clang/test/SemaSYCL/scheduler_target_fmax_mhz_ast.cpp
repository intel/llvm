// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -ast-dump %s | FileCheck %s

// Tests for AST of Intel FPGA scheduler_target_fmax_mhz function attribute.
#include "sycl.hpp"

sycl::queue deviceQueue;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 4
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
[[intel::scheduler_target_fmax_mhz(4)]] void func1() {}

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK_NEXT: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral 6
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void func2() {}

template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void func3() {}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// CHECK: FunctionDecl {{.*}} {{.*}} func4 'void ()'
// CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 10
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}
[[intel::scheduler_target_fmax_mhz(10)]]
[[intel::scheduler_target_fmax_mhz(10)]] void
func4() {}

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
  [[intel::scheduler_target_fmax_mhz(N)]] void operator()() const {
  }
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_1
    // CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

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

    // CHECK-LABEL:  FunctionDecl {{.*}}kernel_name_4
    // CHECK:        SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
    // CHECK-NEXT:   ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:   value: Int 75
    // CHECK-NEXT:   SubstNonTypeTemplateParmExpr {{.*}} 'int'
    // CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 N
    // CHECK-NEXT:   IntegerLiteral {{.*}} 'int' 75
    h.single_task<class kernel_name_4>(
        []() { func3<75>(); });

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
