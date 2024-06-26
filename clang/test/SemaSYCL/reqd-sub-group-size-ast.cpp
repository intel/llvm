// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -ast-dump %s | FileCheck %s

// The test checks AST of [[intel::reqd_sub_group_size()]] attribute.

#include "sycl.hpp"

using namespace sycl;
queue q;

[[intel::reqd_sub_group_size(4)]] void foo() {}

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

class Functor {
public:
  void operator()() const {
    foo();
  }
};

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

// Test that checks template parameter support on function.
// CHECK: FunctionTemplateDecl {{.*}} func
// CHECK: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: TemplateArgument integral 12
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 12
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
template <int N>
[[intel::reqd_sub_group_size(N)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 16
    // CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
    Functor f;
    h.single_task<class kernel_name2>(f);

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 2
    // CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
    h.single_task<class kernel_name3>([]() [[intel::reqd_sub_group_size(2)]] {});

    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name4
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 4
    // CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
    h.single_task<class kernel_name4>([]() [[intel::reqd_sub_group_size(4)]] { foo(); });
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr {{.*}} 'int'
    // CHECK-NEXT: value: Int 6
    // CHECK-NEXT: IntegerLiteral{{.*}}6{{$}}
    h.single_task<class kernel_name5>([]() [[intel::reqd_sub_group_size(6)]] {});

    // CHECK: FunctionDecl {{.*}}kernel_name_6
    // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
    // CHECK-NEXT: ConstantExpr{{.*}}'int'
    // CHECK-NEXT: value: Int 10
    // CHECK-NEXT: SubstNonTypeTemplateParmExpr
    // CHECK-NEXT: NonTypeTemplateParmDecl
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 10
    KernelFunctor<10> f2;
    h.single_task<class kernel_name_6>(f2);

    // Ignore duplicate attribute.
    h.single_task<class kernel_name_7>(
        // CHECK: FunctionDecl {{.*}}kernel_name_7
        // CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
        // CHECK-NEXT: ConstantExpr {{.*}} 'int'
        // CHECK-NEXT: value: Int 8
        // CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
        // CHECK-NOT: IntelReqdSubGroupSizeAttr
        []() [[intel::reqd_sub_group_size(8),
               intel::reqd_sub_group_size(8)]] {});
  });
  func<12>();
  return 0;
}
