// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s
// expected-no-diagnostics

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
// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (MyUnion, int, char, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_y 'char'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_cuda 'float'

// Check kernel inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'MyUnion' 'void (const MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'MyUnion' lvalue ParmVar {{.*}} '_arg_' 'MyUnion'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_x' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'char'
// CHECK-NEXT: DeclRefExpr {{.*}} 'char' lvalue ParmVar {{.*}} '_arg_y' 'char'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} '_arg_cuda' 'float'
