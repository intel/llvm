// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s

// This test checks parameter rewriting for free functions with parameters
// of type scalar, pointer, simple struct and struct with pointers.

#include "sycl.hpp"

struct Simple {
  int x;
  char c[100];
  float f;
};
struct WithPointer {
  int x;
  float* fp;
  float f;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]]
void ff_2(int *ptr, int start, int end, struct Simple S, struct WithPointer P) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + S.x + S.f + S.c[2]+ P.x + 66;
}
// CHECK: FunctionDecl {{.*}} __free_function_ff_2 'void (__global int *, int, int, struct Simple, __generated_WithPointer)'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_ptr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_start 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_end 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_S 'struct Simple':'Simple'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_P '__generated_WithPointer'


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void iotar3(int* usmPtr, struct WithPointer P) {
  usmPtr[55] = P.x + 66;
}
// CHECK: FunctionDecl {{.*}} __free_function_iotar3 'void (__global int *, __generated_WithPointer)'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_usmPtr '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} _arg_P '__generated_WithPointer'