// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// a struct-with-an-array-of-unions and a array-of-struct-with-a-union.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

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
    int d;
  } struct_mem;

  a_kernel<class kernel_A>(
      [=]() {
        int local = union_mem.struct_mem.a[2];
      });

  a_kernel<class kernel_B>(
      [=]() {
        int local = struct_mem.union_mem.a[2];
      });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union MyUnion, int, int, int, float, char, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'union MyUnion':'MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_b 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_c 'char'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_d 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyUnion' 'void (const MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyUnion' lvalue ParmVar {{.*}} '_arg_' 'union MyUnion':'MyUnion'
// CHECK-NEXT: InitListExpr {{.*}} 'MyUnion::MyStruct'
// CHECK-NEXT: InitListExpr {{.*}} 'int [3]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} '_arg_b' 'float'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'char' lvalue ParmVar {{.*}} '_arg_c' 'char'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_d' 'int'

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (union MyUnion, int, int, int, float, char, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_b 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_c 'char'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_d 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' 'void (const MyStruct::MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyStruct::MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' lvalue ParmVar {{.*}} '_arg_union_mem' 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: InitListExpr {{.*}} 'int [3]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} '_arg_b' 'float'
// CHECK-NEXT: InitListExpr {{.*}} 'MyStruct'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'char' lvalue ParmVar {{.*}} '_arg_c' 'char'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_d' 'int'
