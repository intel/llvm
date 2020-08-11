// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// unions containing Arrays.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

template <typename T>
union S {
  T a[3];
};

int main() {

  union union_acc_t {
    int member_acc[2];
  } union_acc;

  S<int> s;

  union foo_inner {
    int foo_inner_x;
    int foo_inner_y;
    int foo_inner_z[2];
  };

  union foo {
    int foo_a;
    foo_inner foo_b[2];
    int foo_c;
  };

  foo union_array[2];

  a_kernel<class kernel_A>(
      [=]() {
        int array = union_acc.member_acc[1];
      });

  a_kernel<class kernel_B>(
      [=]() {
        foo local = union_array[1];
      });

  a_kernel<class kernel_C>(
      [=]() {
        int local = s.a[2];
      });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union union_acc_t, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'union union_acc_t':'union_acc_t'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'int'
// Check kernel_A inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK: InitListExpr {{.*}} 'int [2]'
// CHECK: ImplicitCastExpr
// CHECK: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_member_acc' 'int'
// CHECK: ImplicitCastExpr
// CHECK: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_member_acc' 'int'

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_c 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_c 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit 
// CHECK-NEXT: InitListExpr {{.*}}

// Initializer for first element of inner union array
// CHECK-NEXT: ImplicitCastExpr
// CHECK: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'

// Initializer for second element of inner union array
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// Initializer for union array inside foo i.e. foo_inner foo_b[2]
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_c' 'int'

// Initializer for first element of union_array
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_a' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// Initializer for union array i.e. foo union_array[2]
// CHECK: InitListExpr {{.*}} 'foo [2]
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int [2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_z' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_c' 'int'

// Check kernel_C parameters
// CHECK: FunctionDecl {{.*}}kernel_C{{.*}} 'void (S<int>, int, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'S<int>':'S<int>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int':'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int':'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int':'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'S<int>':'S<int>' 'void (const S<int> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}}  'const S<int>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'S<int>':'S<int>' lvalue ParmVar {{.*}} '_arg_' 'S<int>':'S<int>'
// CHECK-NEXT: InitListExpr {{.*}} 'int [3]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int':'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int':'int' lvalue ParmVar {{.*}} '_arg_a' 'int':'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int':'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int':'int' lvalue ParmVar {{.*}} '_arg_a' 'int':'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int':'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int':'int' lvalue ParmVar {{.*}} '_arg_a' 'int':'int'
