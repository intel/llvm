// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s
// This test checks parameter rewriting for free functions with parameters
// of type scalar, pointer and non-decomposed struct.

#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel_{{.*}} 'void (__global int *, int, int)'
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

// Explicit instantiation with "int*"
template void ff_3(int* ptr, int start, int end);

// CHECK: FunctionDecl {{.*}}__sycl_kernel_{{.*}} 'void (__global int *, int, int)'
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

struct NoPointers {
  int f;
};

struct Pointers {
  int * a;
  float * b;
};

struct Agg {
  NoPointers F1;
  int F2;
  int *F3;
  Pointers F4;
};

struct Agg1 {
  NoPointers F1;
  int F2;
};

struct Derived : Agg {
  int a;
};

class Derived1 : Pointers {
  int a;
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_4(NoPointers S1, Pointers S2, Agg S3) {
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (NoPointers, __generated_Pointers, __generated_Agg)'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S1 'NoPointers'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S2 '__generated_Pointers'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S3 '__generated_Agg'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(NoPointers, Pointers, Agg)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (NoPointers, Pointers, Agg)' lvalue Function {{.*}} 'ff_4' 'void (NoPointers, Pointers, Agg)'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'NoPointers' 'void (const NoPointers &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const NoPointers' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'NoPointers' lvalue ParmVar {{.*}} '__arg_S1' 'NoPointers'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Pointers' 'void (const Pointers &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Pointers' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Pointers' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Pointers *' reinterpret_cast<Pointers *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Pointers *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Pointers' lvalue ParmVar {{.*}} '__arg_S2' '__generated_Pointers'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Agg' 'void (const Agg &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Agg' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Agg' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Agg *' reinterpret_cast<Agg *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Agg *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Agg' lvalue ParmVar {{.*}} '__arg_S3' '__generated_Agg'

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_5(Agg1 S1, Derived S2, Derived1 S3) {
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (Agg1, __generated_Derived, __generated_Derived1)'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S1 'Agg1'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S2 '__generated_Derived'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S3 '__generated_Derived1'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr
// CHECK-NEXT: ImplicitCastExpr{{.*}}'void (*)(Agg1, Derived, Derived1)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr{{.*}}'void (Agg1, Derived, Derived1)' lvalue Function {{.*}} 'ff_5' 'void (Agg1, Derived, Derived1)'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Agg1' 'void (const Agg1 &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Agg1' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'Agg1' lvalue ParmVar {{.*}} '__arg_S1' 'Agg1'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Derived' 'void (const Derived &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Derived' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Derived' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Derived *' reinterpret_cast<Derived *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Derived *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Derived' lvalue ParmVar {{.*}} '__arg_S2' '__generated_Derived'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Derived1' 'void (const Derived1 &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Derived1' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Derived1' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Derived1 *' reinterpret_cast<Derived1 *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Derived1 *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Derived1' lvalue ParmVar {{.*}} '__arg_S3' '__generated_Derived1'

template <typename T1, typename T2>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
  void ff_6(T1 S1, T2 S2, int end) {
}

// Explicit instantiation.
template void ff_6(Agg S1, Derived1 S2, int);
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (__generated_Agg, __generated_Derived1, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S1 '__generated_Agg'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_S2 '__generated_Derived1'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_end 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(Agg, Derived1, int)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (Agg, Derived1, int)' lvalue Function {{.*}} 'ff_6' 'void (Agg, Derived1, int)'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Agg' 'void (const Agg &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Agg' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Agg' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Agg *' reinterpret_cast<struct Agg *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Agg *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Agg' lvalue ParmVar {{.*}} '__arg_S1' '__generated_Agg'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Derived1' 'void (const Derived1 &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const Derived1' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'Derived1' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'Derived1 *' reinterpret_cast<class Derived1 *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_Derived1 *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_Derived1' lvalue ParmVar {{.*}} '__arg_S2' '__generated_Derived1'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '__arg_end' 'int'
