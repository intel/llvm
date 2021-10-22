// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// unions containing Arrays.

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
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union union_acc_t)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_acc 'union union_acc_t':'union_acc_t'

// Check kernel_A inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .union_acc
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_union_acc'
// CHECK-NEXT:  IntegerLiteral {{.*}} 8

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_s 'S<int>':'S<int>'

// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .s
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_s'
// CHECK-NEXT:  IntegerLiteral {{.*}} 12
