// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s
// UNSUPPORTED: system-windows
// This test checks parameter rewriting for free functions with parameters
// of type scalar, pointer, simple struct and struct with pointers.
// Windows support will be added later.

#include "sycl.hpp"

struct Simple {
  int x;
  char c[100];
  float f;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_2(int *ptr, int start, int end, struct Simple S) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + S.x + S.f + S.c[2];
}
// CHECK: FunctionDecl {{.*}} __free_function_ff_2 'void (__global int *, int, int, struct Simple)'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_ptr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_start 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_end 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_S 'struct Simple':'Simple'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int *, int, int, struct Simple)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int *, int, int, struct Simple)' Function {{.*}} 'ff_2' 'void (int *, int, int, struct Simple)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_ptr' '__global int *'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} '_arg_start' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} '_arg_end' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct Simple':'Simple' ParmVar {{.*}} '_arg_S' 'struct Simple':'Simple'


// Templated free function definition.
template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
  void ff_3(T* ptr, T start, int end, struct Simple S) {
    for (int i = start; i <= end; i++)
      ptr[i] = start + S.x + S.f + S.c[2];
}

// Explicit instantiation with �int*�
template void ff_3(int* ptr, int start, int end, struct Simple S);

// CHECK: FunctionDecl {{.*}} __free_function_Z4ff_3IiEvPT_S0_i6Simple 'void (__global int *, int, int, struct Simple)'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_ptr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_start 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_end 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_S 'struct Simple':'Simple'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int *, int, int, struct Simple)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int *, int, int, struct Simple)' Function {{.*}} 'ff_3' 'void (int *, int, int, struct Simple)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_ptr' '__global int *'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} '_arg_start' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' ParmVar {{.*}} '_arg_end' 'int'
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct Simple':'Simple' ParmVar {{.*}} '_arg_S' 'struct Simple':'Simple'
