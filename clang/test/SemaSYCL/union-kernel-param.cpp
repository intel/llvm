// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// union without array.

union MyUnion {
  int x;
  char y;
  float cuda;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  MyUnion accel;

  a_kernel<class kernel>(
      [=]() {
        float local = accel.cuda;
      });
}

// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (MyUnion)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_accel 'MyUnion'

// Check kernel inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'MyUnion' 'void (const MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'MyUnion' lvalue ParmVar {{.*}} '_arg_accel' 'MyUnion'
