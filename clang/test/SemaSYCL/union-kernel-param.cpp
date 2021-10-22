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
// CHECK-NEXT: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .accel
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_accel'
// CHECK-NEXT:  IntegerLiteral {{.*}} 4
