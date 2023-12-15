// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks inheritance support for struct types with accessors
// passed as kernel arguments, which are decomposed to individual fields.

#include "sycl.hpp"

sycl::queue myQueue;

struct AccessorBase {
  int A, B;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField;
};

struct AccessorDerived : AccessorBase,
                         sycl::accessor<char, 1, sycl::access::mode::read> {
  int C;
};

int main() {
  AccessorDerived DerivedObject;
  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel>(
        [=] {
          DerivedObject.use();
        });
  });

  return 0;
}

// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (int, int, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int)'
// CHECK: ParmVarDecl{{.*}} used _arg_A 'int'
// CHECK: ParmVarDecl{{.*}} used _arg_B 'int'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'sycl::id<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base '__global char *'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'sycl::id<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_C 'int'

// Check lambda initialization
// CHECK: VarDecl {{.*}} used __SYCLKernel '(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: InitListExpr {{.*}}
// CHECK-NEXT: InitListExpr {{.*}} 'AccessorDerived'
// CHECK-NEXT: InitListExpr {{.*}} 'AccessorBase'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_A' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_B' 'int'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>' 'void () noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_C' 'int'

// Check __init calls
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} .__init
// CHECK-NEXT: MemberExpr {{.*}} .AccField
// CHECK-NEXT: ImplicitCastExpr {{.*}}'AccessorBase' lvalue <DerivedToBase (AccessorBase)>
// CHECK-NEXT: MemberExpr {{.*}}'AccessorDerived' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}}'(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg_AccField' '__global char *'

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr{{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr{{.*}}'AccessorDerived' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg__base' '__global char *'
