// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck --check-prefixes=CHECK,GEN-AS %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump -fsycl-force-global-as-in-kernel-args %s | FileCheck --check-prefixes=CHECK,GLOB-AS %s

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
// CHECK: FunctionDecl {{.*}}kernel_const{{.*}} 'void (const int) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}} used _arg_some_const 'const int'

// Check that lambda field of const built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const int' lvalue ParmVar {{.*}} '_arg_some_const' 'const int'

// Check kernel parameters
// CHECK: {{.*}}kernel_int{{.*}} 'void (int) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}} used _arg_data 'int'

// Check that lambda field of built-in type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_data' 'int'

// Check kernel parameters
// GLOB-AS: {{.*}}kernel_struct{{.*}} 'void (__generated_test_struct) __attribute__((device_kernel))'
// GEN-AS: {{.*}}kernel_struct{{.*}} 'void (test_struct) __attribute__((device_kernel))'
// GLOB-AS: ParmVarDecl {{.*}} used _arg_s '__generated_test_struct'
// GEN-AS: ParmVarDecl {{.*}} used _arg_s 'test_struct'

// Check that lambda field of struct type is initialized
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}built-in-type-kernel-arg.cpp{{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'test_struct' 'void (const test_struct &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const test_struct' lvalue <NoOp>
// GLOB-AS-NEXT: UnaryOperator {{.*}} 'test_struct' lvalue prefix '*' cannot overflow
// GLOB-AS-NEXT: CXXReinterpretCastExpr {{.*}} 'test_struct *' reinterpret_cast<test_struct *> <BitCast>
// GLOB-AS-NEXT: UnaryOperator {{.*}} '__generated_test_struct *' prefix '&' cannot overflow
// GLOB-AS-NEXT: DeclRefExpr {{.*}} '__generated_test_struct' lvalue ParmVar {{.*}} '_arg_s'
// GEN-AS-NEXT: DeclRefExpr {{.*}} 'test_struct' lvalue ParmVar {{.*}} '_arg_s'

// Check kernel parameters
// CHECK: {{.*}}kernel_pointer{{.*}} 'void (__global int *, __global int *, __wrapper_class) __attribute__((device_kernel))'
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
// GLOB-AS: InitListExpr {{.*}} 'int *[2]'
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// GLOB-AS-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// GLOB-AS-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// GLOB-AS-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array'
// GLOB-AS-NEXT: IntegerLiteral {{.*}} 0
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// GLOB-AS-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// GLOB-AS-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// GLOB-AS-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// GLOB-AS-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array'
// GLOB-AS-NEXT: IntegerLiteral {{.*}} 1
// GEN-AS-NEXT: ArrayInitLoopExpr {{.*}} 'int *[2]'
// GEN-AS-NEXT: OpaqueValueExpr {{.*}} 'int *[2]' lvalue
// GEN-AS-NEXT: MemberExpr {{.*}} 'int *[2]' lvalue .
// GEN-AS-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ptr_array'

// GLOB-AS: FunctionDecl {{.*}}kernel_nns{{.*}} 'void (__generated_test_struct_simple) __attribute__((device_kernel))'
// GEN-AS: FunctionDecl {{.*}}kernel_nns{{.*}} 'void (Nested::TDS) __attribute__((device_kernel))'
// GLOB-AS-NEXT: ParmVarDecl {{.*}} used _arg_tds '__generated_test_struct_simple'
// GEN-AS-NEXT: ParmVarDecl {{.*}} used _arg_tds 'Nested::TDS':'test_struct_simple'

// CHECK: VarDecl {{.*}} used __SYCLKernel
// CHECK: InitListExpr
// CHECK: CXXConstructExpr {{.*}} 'Nested::TDS':'test_struct_simple' 'void (const test_struct_simple &) noexcept'
// CHECK: ImplicitCastExpr {{.*}} 'const test_struct_simple' lvalue <NoOp>
// GEN-AS-NEXT: DeclRefExpr {{.*}} 'Nested::TDS':'test_struct_simple' lvalue ParmVar {{.*}} '_arg_tds'
// GLOB-AS-NEXT: UnaryOperator {{.*}} 'Nested::TDS':'test_struct_simple' lvalue prefix '*' cannot overflow
// GLOB-AS-NEXT: CXXReinterpretCastExpr {{.*}} 'Nested::TDS *' reinterpret_cast<Nested::TDS *> <BitCast>
// GLOB-AS-NEXT: UnaryOperator {{.*}} '__generated_test_struct_simple *' prefix '&' cannot overflow
// GLOB-AS-NEXT: DeclRefExpr {{.*}} '__generated_test_struct_simple' lvalue ParmVar {{.*}} '_arg_tds' '__generated_test_struct_simple'
