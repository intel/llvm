// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

struct Base {
  int A, B;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> AccField;
};

struct Captured : Base,
                  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> {
  int C;
};

int main() {
  Captured Obj;
  cl::sycl::kernel_single_task<class kernel>(
      [=]() {
        Obj.use();
      });
}

// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (int, int, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global char *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, int)'
// CHECK: ParmVarDecl{{.*}} used _arg_A 'int'
// CHECK: ParmVarDecl{{.*}} used _arg_B 'int'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'cl::sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'cl::sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_AccField 'cl::sycl::id<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base '__global char *'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'cl::sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'cl::sycl::range<1>'
// CHECK: ParmVarDecl{{.*}} used _arg__base 'cl::sycl::id<1>'
// CHECK: ParmVarDecl{{.*}} used _arg_C 'int'

// Check lambda initialization
// CHECK: VarDecl {{.*}} used '(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: InitListExpr {{.*}}
// CHECK-NEXT: InitListExpr {{.*}} 'Captured'
// CHECK-NEXT: InitListExpr {{.*}} 'Base'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_A' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_B' 'int'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::accessor<char, 1, cl::sycl::access::mode::read>':'cl::sycl::accessor<char, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::accessor<char, 1, cl::sycl::access::mode::read>':'cl::sycl::accessor<char, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>' 'void () noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_C' 'int'

// Check __init calls
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} .__init
// CHECK-NEXT: MemberExpr {{.*}} .AccField
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Base' lvalue <DerivedToBase (Base)>
// CHECK-NEXT: MemberExpr {{.*}} 'Captured' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}}'(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg_AccField' '__global char *'

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr{{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr{{.*}}'Captured' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}accessor_inheritance.cpp
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg__base' '__global char *'
