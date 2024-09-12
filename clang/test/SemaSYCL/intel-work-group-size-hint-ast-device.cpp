// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Test for AST of work_group_size_hint kernel attribute in SYCL.

#include "sycl.hpp"

using namespace sycl;
queue q;

// Test that checks template parameter support on member function of class template.
template <int SIZE, int SIZE1, int SIZE2>
class KernelFunctor {
public:
  [[sycl::work_group_size_hint(SIZE, SIZE1, SIZE2)]] void operator()() {}
};

void test() {
  KernelFunctor<16, 1, 1>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLWorkGroupSizeHintAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 16
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}

// Test that checks template parameter support on function.
template <int N, int N1, int N2>
[[sycl::work_group_size_hint(N, N1, N2)]] void func3() {}

int check() {
  func3<8, 8, 8>();
  return 0;
}
// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: FunctionDecl {{.*}} {{.*}} used func3 'void ()'
// CHECK: SYCLWorkGroupSizeHintAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 8
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}

class Functor16 {
public:
  [[sycl::work_group_size_hint(16)]] void operator()() const {}
};

class Functor32x16x8 {
public:
  [[sycl::work_group_size_hint(32, 16, 8)]] void operator()() const {}
};

class FunctorAttr {
public:
  [[sycl::work_group_size_hint(128, 256, 512)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: SYCLWorkGroupSizeHintAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
    // CHECK: SYCLWorkGroupSizeHintAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 32
    // CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 8
    // CHECK-NEXT:  IntegerLiteral{{.*}}8{{$}}
    Functor32x16x8 f32x16x8;
    h.single_task<class kernel_name3>(f32x16x8);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
    // CHECK: SYCLWorkGroupSizeHintAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 256
    // CHECK-NEXT:  IntegerLiteral{{.*}}256{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 512
    // CHECK-NEXT:  IntegerLiteral{{.*}}512{{$}}
    FunctorAttr fattr;
    h.single_task<class kernel_name4>(fattr);

  });
  return 0;
}
