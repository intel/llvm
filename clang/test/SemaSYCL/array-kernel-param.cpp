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
// CHECK-NEXT:  DeclStmt
// CHECK-NEXT:   VarDecl {{.*}} __wrapper_union
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .Array
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_Array'
// CHECK-NEXT: IntegerLiteral {{.*}} 8


// Check Kernel_Array_Ptrs parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_Ptrs{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__wrapper_class'
// Check Kernel_Array_Ptrs inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT:  DeclStmt
// CHECK-NEXT:   VarDecl {{.*}} __wrapper_union
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .ArrayOfPointers
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 16

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
// CHECK-NEXT:  VarDecl {{.*}} __wrapper_union
// Init first accessor
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    ArraySubscriptExpr
// CHECK-NEXT:     ImplicitCastExpr
// CHECK-NEXT:      MemberExpr {{.*}} .member_acc
// CHECK-NEXT:       MemberExpr {{.*}} .StructAccArrayObj
// CHECK-NEXT:        MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:         DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:    IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:  MemberExpr {{.*}}__init
// Init second accessor
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:      MemberExpr {{.*}} .member_acc
// CHECK-NEXT:       MemberExpr {{.*}} .StructAccArrayObj
// CHECK-NEXT:        MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:         DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT:  MemberExpr {{.*}}__init

// Check Kernel_TemplatedStructArray parameters
// CHECK: FunctionDecl {{.*}}Kernel_TemplatedStructArray{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_s 'S<int>':'S<int>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT:  VarDecl
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} 'S<int>':'S<int>'
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_s'
// CHECK-NEXT: IntegerLiteral {{.*}} 12

// Check Kernel_Array_2D parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_2D{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_array_2D '__wrapper_class'
// Check Kernel_Array_2D inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT:  VarDecl
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .array_2D
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_array_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 24

// Check Kernel_NonDecomposedStruct parameters.
// CHECK: FunctionDecl {{.*}}Kernel_NonDecomposedStruct{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_NonDecompStructArray '__wrapper_class'
// Check Kernel_NonDecomposedStruct inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT:  VarDecl
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .NonDecompStructArray
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_NonDecompStructArray'
// CHECK-NEXT: IntegerLiteral {{.*}} 32

// Check Kernel_StructWithPointers parameters.
// CHECK: FunctionDecl {{.*}}Kernel_StructWithPointers{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_StructWithPointersArray '__wrapper_class'
// Check Kernel_StructWithPointers inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT:  VarDecl
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .StructWithPointersArray
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_StructWithPointersArray'
// CHECK-NEXT: IntegerLiteral {{.*}} 48

// Check Kernel_Array_Ptrs_2D parameters
// CHECK: FunctionDecl {{.*}}Kernel_Array_Ptrs_2D{{.*}} 'void (__wrapper_class, __wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ArrayOfPointers '__wrapper_class'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT:  VarDecl
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .ArrayOfPointers_2D
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_ArrayOfPointers_2D'
// CHECK-NEXT: IntegerLiteral {{.*}} 48
// CHECK-NEXT: CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .ArrayOfPointers
// CHECK-NEXT:     MemberExpr {{.*}} '(lambda at
// CHECK-NEXT:      DeclRefExpr {{.*}} '__wrapper_union'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT:  UnaryOperator
// CHECK-NEXT:   MemberExpr
// CHECK-NEXT:    DeclRefExpr {{.*}} ParmVar {{.*}} '_arg_ArrayOfPointers'
// CHECK-NEXT: IntegerLiteral {{.*}} 16
