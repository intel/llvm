// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// a struct-with-an-array-of-unions and a array-of-struct-with-a-union.

#include "sycl.hpp"

sycl::queue myQueue;

int main() {

  union MyUnion {
    struct MyStruct {
      int a[3];
      float b;
      char c;
    } struct_mem;
    int d;
  } union_mem;

  struct MyStruct {
    union MyUnion {
      int a[3];
      float b;
      char c;
    } union_mem;
    sycl::accessor<char, 1, sycl::access::mode::read> AccField;
  } struct_mem;

  struct MyStructWithPtr {
    union MyUnion {
      int a[3];
      float b;
      char c;
    } union_mem;
    int *d;
  } structWithPtr_mem;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_A>(
        [=]() {
          int local = union_mem.struct_mem.a[2];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_B>(
        [=]() {
          int local = struct_mem.union_mem.a[2];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_C>(
        [=]() {
          int local = structWithPtr_mem.union_mem.a[2];
        });
  });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union MyUnion)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyUnion'

// Check kernel_A inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .union_mem
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_union_mem'
// CHECK-NEXT:  IntegerLiteral {{.*}} 20

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (union MyUnion, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField 'sycl::id<1>'

// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .union_mem
// CHECK-NEXT:     MemberExpr {{.*}} .struct_mem
// CHECK-NEXT:      MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:       DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_union_mem'
// CHECK-NEXT:  IntegerLiteral {{.*}} 12
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .AccField
// CHECK-NEXT:      MemberExpr {{.*}} .struct_mem
// CHECK-NEXT:       MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:        DeclRefExpr {{.*}} '__wrapper_union'

// Check call to __init to initialize AccField
// CHECK-NEXT: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} lvalue .AccField
// CHECK-NEXT: MemberExpr {{.*}} lvalue .struct_mem
// CHECK-NEXT: MemberExpr {{.*}} '(lambda at
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_union'

// Check kernel_C parameters
// CHECK: FunctionDecl {{.*}}kernel_C{{.*}} 'void (__generated_MyStructWithPtr)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_structWithPtr_mem '__generated_MyStructWithPtr'

// Check kernel_C inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .structWithPtr_mem
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_structWithPtr_mem'
// CHECK-NEXT:  IntegerLiteral {{.*}} 24
