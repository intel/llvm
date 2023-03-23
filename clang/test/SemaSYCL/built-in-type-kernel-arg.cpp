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
// CHECK:      VarDecl {{.*}} '__wrapper_union'
// CHECK:      BinaryOperator {{.*}} '='
// CHECK-NEXT:  MemberExpr {{.*}} .some_const
// CHECK-NEXT:   MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:    DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_some_const'

// Check kernel parameters
// CHECK: {{.*}}kernel_int{{.*}} 'void (int)'
// CHECK: ParmVarDecl {{.*}} used _arg_data 'int'

// Check that lambda field of built-in type is initialized
// CHECK: VarDecl {{.*}} '__wrapper_union'
// CHECK:      BinaryOperator {{.*}} '='
// CHECK-NEXT:  MemberExpr {{.*}} .data
// CHECK-NEXT:   MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:    DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_data'

// Check kernel parameters
// CHECK: {{.*}}kernel_struct{{.*}} 'void (__generated_test_struct)'
// CHECK: ParmVarDecl {{.*}} used _arg_s '__generated_test_struct'

// Check that lambda field of struct type is initialized
// CHECK: VarDecl {{.*}} '__wrapper_union'
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .s
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_s'
// CHECK-NEXT: IntegerLiteral {{.*}} 80

// Check kernel parameters
// CHECK: {{.*}}kernel_pointer{{.*}} 'void (__global int *, __global int *, __wrapper_class)'
// CHECK: ParmVarDecl {{.*}} used _arg_new_data_addr '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_data_addr '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array '__wrapper_class'
// CHECK: VarDecl {{.*}} '__wrapper_union'

// Check that lambda fields of pointer types are initialized
// CHECK:      BinaryOperator {{.*}} '='
// CHECK-NEXT:  MemberExpr {{.*}} .new_data_addr
// CHECK-NEXT:   MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:    DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion> 
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_new_data_addr'
// CHECK-NEXT:  BinaryOperator {{.*}} '='
// CHECK-NEXT:   MemberExpr {{.*}} .data_addr
// CHECK-NEXT:    MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:     DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:   ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion> 
// CHECK-NEXT:    ImplicitCastExpr
// CHECK-NEXT:      DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_data_addr'
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .ptr_array
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr
// CHECK-NEXT:     DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_ptr_array'
// CHECK-NEXT:  IntegerLiteral {{.*}} 16

// CHECK: FunctionDecl {{.*}}kernel_nns{{.*}} 'void (__generated_test_struct_simple)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_tds '__generated_test_struct_simple'

// CHECK: VarDecl {{.*}} __wrapper_union
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .tds
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_tds'
// CHECK-NEXT:  IntegerLiteral {{.*}} 16
