// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct initialization for arguments
// that have struct or built-in type inside the OpenCL kernel

#include "sycl.hpp"

sycl::queue deviceQueue;

struct test_struct {
  int data;
  int *ptr;
  int *ptr_array1[2];
  int *ptr_array2[2][3];
};

void test(const int some_const) {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_const>(
        [=]() {
          int a = some_const;
        });
  });
}

struct test_struct_simple {
  int data;
  int *ptr;
};

struct Nested {
typedef test_struct_simple TDS;
};

int main() {
  int data = 5;
  int* data_addr = &data;
  int* new_data_addr = nullptr;
  int *ptr_array[2];
  test_struct s;
  s.data = data;

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_int>(
        [=]() {
          int kernel_data = data;
        });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_struct>(
        [=]() {
          test_struct k_s;
          k_s = s;
        });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_pointer>(
        [=]() {
          new_data_addr[0] = data_addr[0];
          int *local = ptr_array[1];
        });
  });

  Nested::TDS tds;
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_nns>(
        [=]() {
          test_struct_simple k_s;
          k_s = tds;
        });
  });

  const int some_const = 10;
  test(some_const);
  return 0;
}
// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel_const{{.*}} 'void (const int)'
// CHECK: ParmVarDecl {{.*}} used _arg_some_const 'const int'

// Check that lambda field of const built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const int' lvalue ParmVar {{.*}} '_arg_some_const' 'const int'

// Check kernel parameters
// CHECK: {{.*}}kernel_int{{.*}} 'void (int)'
// CHECK: ParmVarDecl {{.*}} used _arg_data 'int'

// Check that lambda field of built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_data' 'int'

// Check kernel parameters
// CHECK: {{.*}}kernel_struct{{.*}} 'void (__generated_test_struct)'
// CHECK: ParmVarDecl {{.*}} used _arg_s '__generated_test_struct'

// Check that lambda field of struct type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'test_struct' 'void (const test_struct &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const test_struct' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'test_struct' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'test_struct *' reinterpret_cast<test_struct *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_test_struct *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_test_struct' lvalue ParmVar {{.*}} '_arg_s'

// Check kernel parameters
// CHECK: {{.*}}kernel_pointer{{.*}} 'void (__global int *, __global int *, __wrapper_class)'
// CHECK: ParmVarDecl {{.*}} used _arg_new_data_addr '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_data_addr '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array '__wrapper_class'
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'

// Check that lambda fields of pointer types are initialized
// CHECK: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_new_data_addr' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_data_addr' '__global int *'
// CHECK: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array'
// CHECK-NEXT: IntegerLiteral {{.*}} 1

// CHECK: FunctionDecl {{.*}}kernel_nns{{.*}} 'void (__generated_test_struct_simple)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_tds '__generated_test_struct_simple'

// CHECK: VarDecl {{.*}} used __SYCLKernel
// CHECK: InitListExpr
// CHECK: CXXConstructExpr {{.*}} 'Nested::TDS':'test_struct_simple' 'void (const test_struct_simple &) noexcept'
// CHECK: ImplicitCastExpr {{.*}} 'const test_struct_simple' lvalue <NoOp>
// CHECK: UnaryOperator {{.*}} 'Nested::TDS':'test_struct_simple' lvalue prefix '*' cannot overflow
// CHECK: CXXReinterpretCastExpr {{.*}} 'Nested::TDS *' reinterpret_cast<struct Nested::TDS *> <BitCast>
// CHECK: UnaryOperator {{.*}} '__generated_test_struct_simple *' prefix '&' cannot overflow
// CHECK: DeclRefExpr {{.*}} '__generated_test_struct_simple' lvalue ParmVar {{.*}} '_arg_tds' '__generated_test_struct_simple'
