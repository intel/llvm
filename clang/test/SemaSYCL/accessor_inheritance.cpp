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
// CHECK: VarDecl {{.*}} used __wrapper_union '__wrapper_union'

// Init A
// memcpy(lambda.A, __arg_A, 4)
//
// CHECK: BinaryOperator {{.*}} '='
// CHECK:  MemberExpr {{.*}} 'int' lvalue .A
// CHECK:   ImplicitCastExpr {{.*}} <DerivedToBase (AccessorBase)>
// CHECK:    MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived' lvalue .
// CHECK:     MemberExpr {{.*}} '(lambda at
// CHECK:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK:  DeclRefExpr {{.*}} '_arg_A'

// Init B
// memcpy(lambda.B, __arg_B, 4)
//
// CHECK: BinaryOperator {{.*}} '='
// CHECK:  MemberExpr {{.*}} 'int' lvalue .B
// CHECK:   ImplicitCastExpr {{.*}} <DerivedToBase (AccessorBase)>
// CHECK:    MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived' lvalue .
// CHECK:     MemberExpr {{.*}} '(lambda at
// CHECK:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK:  DeclRefExpr {{.*}} '_arg_B'

// Init AccField
// placement new
//
// CHECK: CXXNewExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read> *' global Function {{.*}} 'operator new'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>' 'void () noexcept'
// CHECK: MemberExpr {{.*}}'sycl::accessor<char, 1, sycl::access::mode::read>':'sycl::accessor<char, 1, sycl::access::mode::read>' lvalue .AccField
// CHECK-NEXT: ImplicitCastExpr {{.*}} <DerivedToBase (AccessorBase)>
// CHECK-NEXT:  MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived'
// CHECK-NEXT:   MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:    DeclRefExpr {{.*}} '__wrapper_union'

// call to __init
//
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (PtrType, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT:  MemberExpr {{.*}} .AccField {{.*}}
// CHECK-NEXT:   ImplicitCastExpr {{.*}} <DerivedToBase (AccessorBase)>
// CHECK-NEXT:    MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived'
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK:       DeclRefExpr {{.*}} '_arg_AccField' '__global char *'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_AccField' 'sycl::range<1>'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_AccField' 'sycl::range<1>'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'id<1>':'sycl::id<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_AccField' 'sycl::id<1>'

// Init inherited accessor
// placement new
//
// CHECK:      CXXNewExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read> *' global Function {{.*}} 'operator new'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>' 'void () noexcept'
// CHECK:       ImplicitCastExpr {{.*}} <DerivedToBase (accessor)>
// CHECK-NEXT:   MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived'
// CHECK-NEXT:    MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:     DeclRefExpr {{.*}} '__wrapper_union'

// call to __init
//
// CHECK:      CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (PtrType, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT:  ImplicitCastExpr {{.*}} <DerivedToBase (accessor)>
// CHECK-NEXT:   MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived'
// CHECK-NEXT:    MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:     DeclRefExpr {{.*}} '__wrapper_union'
// CHECK:       DeclRefExpr {{.*}} '_arg__base' '__global char *'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg__base' 'sycl::range<1>'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg__base' 'sycl::range<1>'
// CHECK-NEXT:  CXXConstructExpr {{.*}} 'id<1>':'sycl::id<1>'
// CHECK:        DeclRefExpr {{.*}} ParmVar {{.*}} '_arg__base' 'sycl::id<1>'

// Init C
// memcpy(lambda.C, __arg_C, 4)
//
// CHECK: BinaryOperator {{.*}} '='
// CHECK:  MemberExpr {{.*}} 'int' lvalue .C
// CHECK:   MemberExpr {{.*}} 'AccessorDerived':'AccessorDerived' lvalue .
// CHECK:    MemberExpr {{.*}} '(lambda at
// CHECK:     DeclRefExpr {{.*}} '__wrapper_union'
// CHECK:  DeclRefExpr {{.*}} '_arg_C'
