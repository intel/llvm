// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s
// UNSUPPORTED: system-windows
// This test checks parameter rewriting for free functions with parameters
// of type scalar and pointer.
// Windows support will be added later.

#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}
// CHECK: FunctionDecl {{.*}} _Z18__sycl_kernel_ff_2Piii 'void (__global int *, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_ptr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_start 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_end 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int *, int, int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int *, int, int)' lvalue Function {{.*}} 'ff_2' 'void (int *, int, int)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '__arg_ptr' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '__arg_start' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '__arg_end' 'int'


// Templated free function definition.
template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
  void ff_3(T* ptr, T start, int end) {
    for (int i = start; i <= end; i++)
      ptr[i] = start;
}

// Explicit instantiation with “int*”
template void ff_3(int* ptr, int start, int end);

// CHECK: FunctionDecl {{.*}} _Z18__sycl_kernel_ff_3Piii 'void (__global int *, int, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_ptr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_start 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} __arg_end 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int *, int, int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int *, int, int)' lvalue Function {{.*}} 'ff_3' 'void (int *, int, int)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '__arg_ptr' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '__arg_start' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '__arg_end' 'int'
