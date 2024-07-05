// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -ast-dump %s | FileCheck %s

// Test for AST of reqd_work_group_size kernel attribute in SYCL 1.2.1.

#include "sycl.hpp"

using namespace sycl;
queue q;

// Test that checks template parameter support on member function of class template.
template <int SIZE, int SIZE1, int SIZE2>
class KernelFunctor {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() {}
};

void test() {
  KernelFunctor<16, 1, 1>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLReqdWorkGroupSizeAttr
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
[[sycl::reqd_work_group_size(N, N1, N2)]] void func3() {}

int check() {
  func3<8, 8, 8>();
  return 0;
}
// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: FunctionDecl {{.*}} {{.*}} used func3 'void ()'
// CHECK: SYCLReqdWorkGroupSizeAttr
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

[[sycl::reqd_work_group_size(4)]] void f4() {}

class Functor16 {
public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const {}
};

class Functor16x16x16 {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const {}
};

class Functor {
public:
  void operator()() const {
    f4();
  }
};

class FunctorAttr {
public:
  [[sycl::reqd_work_group_size(128, 128, 128)]] void operator()() const {}
};

// Test of redeclaration of [[intel::max_work_group_size()]] and [[sycl::reqd_work_group_size()]].
[[intel::no_global_work_offset]] void func1();
[[intel::max_work_group_size(4, 4, 4)]] void func1();
[[sycl::reqd_work_group_size(2, 2, 2)]] void func1() {}

[[sycl::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {}

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    Functor f;
    h.single_task<class kernel_name2>(f);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    Functor16x16x16 f16x16x16;
    h.single_task<class kernel_name3>(f16x16x16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 128
    // CHECK-NEXT:  IntegerLiteral{{.*}}128{{$}}
    FunctorAttr fattr;
    h.single_task<class kernel_name4>(fattr);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 32
    // CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 32
    // CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 32
    // CHECK-NEXT:  IntegerLiteral{{.*}}32{{$}}
    h.single_task<class kernel_name5>([]() [[sycl::reqd_work_group_size(32, 32, 32)]] {
      f32x32x32();
    });

    // CHECK:  FunctionDecl {{.*}} {{.*}}kernel_name6
    // CHECK:  SYCLIntelNoGlobalWorkOffsetAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    // CHECK: SYCLIntelMaxWorkGroupSizeAttr {{.*}} Inherited
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK: SYCLReqdWorkGroupSizeAttr
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    h.single_task<class kernel_name6>(
        []() { func1(); });
  });
  return 0;
}
