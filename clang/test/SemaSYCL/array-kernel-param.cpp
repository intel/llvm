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

  struct StructWithAccessors {
    Accessor member_acc[2];
  } StructAccArrayObj;

  S<int> s;

  struct StructWithPointers {
    int x;
    int y;
    int *ArrayOfPtrs[2];
  };

  struct DecomposedStruct {
    int a;
    StructWithPointers SWPtrsMem[2];
    int *Array_2D_Ptrs[2][1];
    int c;
  };

  // Not decomposed.
  struct NonDecomposedStruct {
    int a;
    int Array_2D[2][1];
    int c;
  };

  DecomposedStruct DecompStructArray[2];
  NonDecomposedStruct NonDecompStructArray[2];

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
    h.single_task<class Kernel_DecomposedStruct>(
        [=] {
          DecomposedStruct local = DecompStructArray[1];
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
// CHECK: FunctionDecl {{.*}}Kernel_Array_Ptrs{{.*}} 'void (__global int *, __global int *)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__global int *'
// Check Kernel_Array_Ptrs inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_ArrayOfPointers' '__global int *'

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
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
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

// Check Kernel_DecomposedStruct parameters
// CHECK: FunctionDecl {{.*}}Kernel_DecomposedStruct{{.*}} 'void (int, int, int, __wrapper_class, __wrapper_class, int, int, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, int, int, int, int, __wrapper_class, __wrapper_class, int, int, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_Array_2D_Ptrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_Array_2D_Ptrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_c 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPtrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_Array_2D_Ptrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_Array_2D_Ptrs '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_c 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'

// Initializer for struct array i.e. DecomposedStruct DecompStructArray[2]
// CHECK-NEXT: InitListExpr {{.*}} 'DecomposedStruct[2]'

// Initializer for first element of DecompStructArray
// CHECK-NEXT: InitListExpr {{.*}} 'DecomposedStruct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'

// Initializer for struct array inside DecomposedStruct i.e. StructWithPointers SWPtrsMem[2]
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers[2]'
// Initializer for first element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_x' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// Initializer for second element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][1]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array_2D_Ptrs' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array_2D_Ptrs' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar {{.*}} '_arg_c' 'int'

// Initializer for second element of DecompStructArray
// CHECK-NEXT: InitListExpr {{.*}} 'DecomposedStruct'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_a' 'int'

// Initializer for struct array inside DecomposedStruct i.e. StructWithPointers SWPtrsMem[2]
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers[2]'
// Initializer for first element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_x' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// Initializer for second element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'StructWithPointers'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_ArrayOfPtrs' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][1]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array_2D_Ptrs' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_Array_2D_Ptrs' '__wrapper_class'

// Check Kernel_TemplatedStructArray parameters
// CHECK: FunctionDecl {{.*}}Kernel_TemplatedStructArray{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_s 'S<int>':'S<int>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'S<int>':'S<int>' 'void (const S<int> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'S<int>':'S<int>' lvalue ParmVar {{.*}} '_arg_s' 'S<int>':'S<int>'

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
// CHECK-NEXT: CXXConstructExpr {{.*}} 'NonDecomposedStruct' 'void (const NonDecomposedStruct &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const NonDecomposedStruct' lvalue <NoOp>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'NonDecomposedStruct' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'NonDecomposedStruct *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'NonDecomposedStruct[2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'NonDecomposedStruct[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_NonDecompStructArray' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
