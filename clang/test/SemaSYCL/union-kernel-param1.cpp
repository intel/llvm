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
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union union_acc_t':'union_acc_t' 'void (const union_acc_t &) noexcept'
// CHECK: ImplicitCastExpr {{.*}} 'const union_acc_t'
// CHECK: DeclRefExpr {{.*}} 'union union_acc_t':'union_acc_t' lvalue ParmVar {{.*}} '_arg_union_acc' 'union union_acc_t':'union_acc_t'

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_s 'S<int>'

// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'S<int>' 'void (const S<int> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}}  'const S<int>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'S<int>' lvalue ParmVar {{.*}} '_arg_s' 'S<int>'
