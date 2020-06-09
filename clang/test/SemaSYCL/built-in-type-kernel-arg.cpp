// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct initialization for arguments
// that have struct or built-in type inside the OpenCL kernel

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

struct test_struct {
  int data;
};

void test(const int some_const) {
  kernel<class kernel_const>(
      [=]() {
        int a = some_const;
      });
}

int main() {
  int data = 5;
  test_struct s;
  s.data = data;
  kernel<class kernel_int>(
      [=]() {
        int kernel_data = data;
      });
  kernel<class kernel_struct>(
      [=]() {
        test_struct k_s;
        k_s = s;
      });
  const int some_const = 10;
  test(some_const);
  return 0;
}
// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel_const{{.*}} 'void (const int)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'const int'

// Check that lambda field of const built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const int' lvalue ParmVar {{.*}} '_arg_' 'const int'

// Check kernel parameters
// CHECK: {{.*}}kernel_int{{.*}} 'void (int)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'int'

// Check that lambda field of built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check kernel parameters
// CHECK: {{.*}}kernel_struct{{.*}} 'void (test_struct)'
// CHECK: ParmVarDecl {{.*}} used _arg_ 'test_struct'

// Check that lambda field of struct type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}}'test_struct'{{.*}}void (const test_struct &)
// CHECK-NEXT: ImplicitCastExpr {{.*}}'const test_struct' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'test_struct' lvalue ParmVar {{.*}} '_arg_' 'test_struct'
