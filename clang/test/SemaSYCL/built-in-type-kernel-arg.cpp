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
// CHECK: {{.*}}kernel_struct{{.*}} 'void (int, __wrapper_class, __wrapper_class, __wrapper_class
// CHECK-SAME: __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class)'
// CHECK: ParmVarDecl {{.*}} used _arg_data 'int'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array1 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array1 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'
// CHECK: ParmVarDecl {{.*}} used _arg_ptr_array2 '__wrapper_class'

// Check that lambda field of struct type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'test_struct'{{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_data' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array1' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array1' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][3]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[3]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[3]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}}  '__global int *' lvalue . {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array2' '__wrapper_class'

// Check kernel parameters
// CHECK: {{.*}}kernel_pointer{{.*}} 'void (__global int *, __global int *, __global int *, __global int *)'
// CHECK: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'

// Check that lambda fields of pointer types are initialized
// CHECK: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'
// CHECK: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'
