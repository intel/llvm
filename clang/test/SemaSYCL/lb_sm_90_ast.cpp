// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -triple nvptx-unknown-unknown -target-cpu sm_90 -Wno-c++23-extensions %s | FileCheck %s

// Tests for AST of Intel max_work_group_size, min_work_groups_per_cu and
// max_work_groups_per_mp attribute.

#include "sycl.hpp"

sycl::queue deviceQueue;

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func2
// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxWorkGroupSizeAttr  {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: SYCLIntelMinWorkGroupsPerComputeUnitAttr   {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK-NEXT: SYCLIntelMaxWorkGroupsPerMultiprocessorAttr   {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: TemplateArgument integral '6'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLIntelMaxWorkGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: SYCLIntelMinWorkGroupsPerComputeUnitAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: SYCLIntelMaxWorkGroupsPerMultiprocessorAttr {{.*}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 6
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
template <int N>
[[intel::max_work_group_size(N, 8, 8), intel::min_work_groups_per_cu(N),
  intel::max_work_groups_per_mp(N)]] void
func2() {}

// Test that checks template parameter support on class member function.
template <int N> class KernelFunctor2 {
public:
  [[intel::max_work_group_size(N, 8, 8), intel::min_work_groups_per_cu(N),
    intel::max_work_groups_per_mp(N)]] void
  operator()() const {}
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLIntelMaxWorkGroupSizeAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 8
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 8
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
    // CHECK: SYCLIntelMinWorkGroupsPerComputeUnitAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    // CHECK: SYCLIntelMaxWorkGroupsPerMultiprocessorAttr
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 3
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    KernelFunctor2<3> f2;
    h.single_task<class kernel_name_2>(f2);

    // CHECK-LABEL: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLIntelMaxWorkGroupSizeAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 8
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 8
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 8
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
    // CHECK-NEXT: SYCLIntelMinWorkGroupsPerComputeUnitAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
    // CHECK-NEXT: SYCLIntelMaxWorkGroupsPerMultiprocessorAttr {{.*}}
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 6
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
    h.single_task<class kernel_name_3>(
        [] [[intel::max_work_group_size(8, 8, 8),
             intel::min_work_groups_per_cu(4),
             intel::max_work_groups_per_mp(6)]] () {});
  });

  func2<6>();

  return 0;
}
