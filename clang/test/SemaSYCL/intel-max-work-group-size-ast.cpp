// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 | FileCheck %s

// The test checks support and functionality of [[intel:::max_work_group_size()]] attribute.
#include "sycl.hpp"

using namespace sycl;
queue q;

struct FuncObj {
  [[intel::max_work_group_size(4, 4, 4)]] void operator()() const {}
};

// Test that checks template parameter support on member function of class template.
template <int SIZE, int SIZE1, int SIZE2>
class KernelFunctor {
public:
  [[intel::max_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() {}
};

// Test that checks template parameter support on function.
template <int N, int N1, int N2>
[[intel::max_work_group_size(N, N1, N2)]] void func() {}

int check() {
  // CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
  // CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
  // CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
  // CHECK: SYCLIntelMaxWorkGroupSizeAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 4
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 4
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 4
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
  KernelFunctor<4, 4, 4>();

  // CHECK: FunctionTemplateDecl {{.*}} {{.*}} func
  // CHECK: FunctionDecl {{.*}} {{.*}} used func 'void ()'
  // CHECK: SYCLIntelMaxWorkGroupSizeAttr {{.*}}
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
  // CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
  func<8, 8, 8>();
  return 0;
}

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelMaxWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::max_work_group_size(8, 8, 8)]] {});

    // Ignore duplicate attribute.
    h.single_task<class test_kernel10>(
        // CHECK-LABEL: FunctionDecl {{.*}}test_kernel10
        // CHECK:       SYCLIntelMaxWorkGroupSizeAttr
        // CHECK-NEXT:  ConstantExpr{{.*}}'int'
        // CHECK-NEXT:  value: Int 2
        // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
        // CHECK-NEXT:  ConstantExpr{{.*}}'int'
        // CHECK-NEXT:  value: Int 2
        // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
        // CHECK-NEXT:  ConstantExpr{{.*}}'int'
        // CHECK-NEXT:  value: Int 2
        // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
        // CHECK-NOT:   SYCLIntelMaxWorkGroupSizeAttr
        []() [[intel::max_work_group_size(2, 2, 2),
               intel::max_work_group_size(2, 2, 2)]] {});
  });
  return 0;
}
