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

  a_kernel<class kernel_A>(
      [=]() {
        int array = union_acc.member_acc[1];
      });

  a_kernel<class kernel_B>(
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
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (S<int>, int, int, int)'
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
