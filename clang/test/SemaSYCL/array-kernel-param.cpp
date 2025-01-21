// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// arrays, Accessor arrays, and structs containing Accessors.

#include "sycl.hpp"

sycl::queue myQueue;

using namespace sycl;

template <typename T>
struct S {
  T a[3];
};

int main() {

  using Accessor =
      sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;

  Accessor ReadWriteAccessor[2];
  int Array[2];
  int *ArrayOfPointers[2];
  int *ArrayOfPointers_2D[2][3];

  struct StructWithAccessors {
    Accessor member_acc[2];
  } StructAccArrayObj;

  S<int> s;

  // Not decomposed.
  struct NonDecomposedStruct {
    int a;
    int Array_2D[2][1];
    int c;
  };

  struct StructWithPointers {
    int *Ptr;
    int *ArrayOfPtrs[2];
  };

  NonDecomposedStruct NonDecompStructArray[2];
  StructWithPointers StructWithPointersArray[2];

  int array_2D[2][3];

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_Accessor>(
        [=] {
          ReadWriteAccessor[1].use();
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_Array>(
        [=] {
          int local = Array[1];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_Array_Ptrs>(
        [=] {
          int local = *ArrayOfPointers[1];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_StructAccArray>(
        [=] {
          StructAccArrayObj.member_acc[2].use();
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_TemplatedStructArray>(
        [=] {
          int local = s.a[2];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_Array_2D>(
        [=] {
          int local = array_2D[1][1];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_NonDecomposedStruct>(
        [=] {
          NonDecomposedStruct local = NonDecompStructArray[0];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_StructWithPointers>(
        [=] {
           StructWithPointers local = StructWithPointersArray[0];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_Array_Ptrs_2D>(
        [=] {
          int local1 = *ArrayOfPointers_2D[0][0];
          int local2 = *ArrayOfPointers[0];
        });
  });

}

// Check Kernel_Accessor parameters
// CHECK: FunctionDecl {{.*}}Kernel_Accessor{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor '__global int *'
// CHECK-NEXT: SYCLAccessorPtrAttr
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor '__global int *'
// CHECK-NEXT: SYCLAccessorPtrAttr
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ReadWriteAccessor 'sycl::id<1>'
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init

// Check Kernel_Array parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_Array '__wrapper_class'
// Check Kernel_Array inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int[2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array' '__wrapper_class'

// Check Kernel_Array_Ptrs parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_Ptrs{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__wrapper_class'
// Check Kernel_Array_Ptrs inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// Initializer for ArrayOfPointers[0]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// Initializer for ArrayOfPointers[1]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 1

// Check Kernel_StructAccArray parameters
// CHECK: FunctionDecl {{.*}}Kernel_StructAccArray{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>, __global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: SYCLAccessorPtrAttr
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: SYCLAccessorPtrAttr
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'sycl::id<1>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used __SYCLKernel '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithAccessors'
// CHECK-NEXT: InitListExpr {{.*}} 'Accessor[2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Accessor'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Accessor'

// Check __init functions are called
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init

// Check Kernel_TemplatedStructArray parameters
// CHECK: FunctionDecl {{.*}}Kernel_TemplatedStructArray{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_s 'S<int>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used __SYCLKernel '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'S<int>' 'void (const S<int> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'S<int>' lvalue ParmVar {{.*}} '_arg_s' 'S<int>'

// Check Kernel_Array_2D parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_2D{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_array_2D '__wrapper_class'
// Check Kernel_Array_2D inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int[2][3]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_array_2D' '__wrapper_class'
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int[3]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_array_2D' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int[2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_array_2D' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned

// Check Kernel_NonDecomposedStruct parameters.
// CHECK: FunctionDecl {{.*}}Kernel_NonDecomposedStruct{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_NonDecompStructArray '__wrapper_class'
// Check Kernel_NonDecomposedStruct inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'NonDecomposedStruct[2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'NonDecomposedStruct[2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'NonDecomposedStruct[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_NonDecompStructArray' '__wrapper_class'
// CHECK-NEXT: CXXConstructExpr {{.*}}'NonDecomposedStruct' 'void (const NonDecomposedStruct &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const NonDecomposedStruct' lvalue <NoOp>
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'NonDecomposedStruct' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'NonDecomposedStruct *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'NonDecomposedStruct[2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'NonDecomposedStruct[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_NonDecompStructArray' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned

// Check Kernel_StructWithPointers parameters.
// CHECK: FunctionDecl {{.*}}Kernel_StructWithPointers{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_StructWithPointersArray '__wrapper_class'
// Check Kernel_StructWithPointers inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers[2]'
// Initializer for StructWithPointersArray[0]
// CHECK-NEXT: CXXConstructExpr {{.*}} 'StructWithPointers' 'void (const StructWithPointers &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const StructWithPointers' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'StructWithPointers' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'StructWithPointers *' reinterpret_cast<StructWithPointers *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_StructWithPointers *' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__generated_StructWithPointers' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__generated_StructWithPointers *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__generated_StructWithPointers[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_StructWithPointersArray'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// Initializer for StructWithPointersArray[1]
// CHECK: CXXConstructExpr {{.*}} 'StructWithPointers' 'void (const StructWithPointers &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const StructWithPointers' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'StructWithPointers' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'StructWithPointers *' reinterpret_cast<StructWithPointers *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_StructWithPointers *' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__generated_StructWithPointers' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__generated_StructWithPointers *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__generated_StructWithPointers[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_StructWithPointersArray'
// CHECK-NEXT: IntegerLiteral {{.*}} 1

// Check Kernel_Array_Ptrs_2D parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_Ptrs_2D{{.*}} 'void (__wrapper_class, __wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__wrapper_class'

// Check Kernel_Array_Ptrs_2D inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr

// Initializer for ArrayOfPointers_2D
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][3]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[3]'
// Initializer for ArrayOfPointers_2D[0][0]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// CHECK-NEXT: IntegerLiteral {{.*}} 0

// Initializer for ArrayOfPointers_2D[0][1]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// CHECK-NEXT: IntegerLiteral {{.*}} 1

// Initializer for ArrayOfPointers_2D[0][2]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 0
// CHECK-NEXT: IntegerLiteral {{.*}} 2

// CHECK-NEXT: InitListExpr {{.*}} 'int *[3]'

// Initializer for ArrayOfPointers_2D[1][0]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: IntegerLiteral {{.*}} 0

// Initializer for ArrayOfPointers_2D[1][1]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: IntegerLiteral {{.*}} 1

// Initializer for ArrayOfPointers_2D[1][2]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *[3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *(*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: IntegerLiteral {{.*}} 2

// Initializer for ArrayOfPointers
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// Initializer for ArrayOfPointers[0]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 0

// Initializer for ArrayOfPointers[1]
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: UnaryOperator {{.*}} 'int *' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'int **' reinterpret_cast<int **> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__global int **' prefix '&' cannot overflow
// CHECK-NEXT: ArraySubscriptExpr {{.*}} '__global int *' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int **' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 1
