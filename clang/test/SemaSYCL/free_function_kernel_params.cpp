// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s
// This test checks parameter rewriting for free functions with parameters
// of type scalar, pointer, non-decomposed struct, work group memory, dynamic work group memory 
// and special types.

#include "sycl.hpp"

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel_{{.*}} 'void (__global int *, int, int) __attribute__((device_kernel))'
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

// CHECK: FunctionDecl {{.*}}__sycl_kernel_{{.*}} 'void (__global int *, int, int) __attribute__((device_kernel))'
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
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (NoPointers, __generated_Pointers, __generated_Agg) __attribute__((device_kernel))'
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
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (Agg1, __generated_Derived, __generated_Derived1) __attribute__((device_kernel))'
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
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (__generated_Agg, __generated_Derived1, int) __attribute__((device_kernel))'
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

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_7(sycl::work_group_memory<int> mem) {
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (__local int *) __attribute__((device_kernel))'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_Ptr '__local int *'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used mem 'sycl::work_group_memory<int>' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::work_group_memory<int>' 'void () noexcept'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__local int *)' lvalue .__init
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::work_group_memory<int>' Var {{.*}} 'mem' 'sycl::work_group_memory<int>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__local int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__local int *' lvalue ParmVar {{.*}} '__arg_Ptr' '__local int *'
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(sycl::work_group_memory<int>)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (sycl::work_group_memory<int>)' lvalue Function {{.*}} 'ff_7' 'void (sycl::work_group_memory<int>)'
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::work_group_memory<int>' Var {{.*}} 'mem' 'sycl::work_group_memory<int>'

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_8(sycl::dynamic_work_group_memory<int> DynMem) {
}
// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (__local int *) __attribute__((device_kernel))'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_Ptr '__local int *'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used DynMem 'sycl::dynamic_work_group_memory<int>' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::dynamic_work_group_memory<int>' 'void () noexcept'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__local int *)' lvalue .__init
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::dynamic_work_group_memory<int>' Var {{.*}} 'DynMem' 'sycl::dynamic_work_group_memory<int>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__local int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__local int *' lvalue ParmVar {{.*}} '__arg_Ptr' '__local int *'
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(sycl::dynamic_work_group_memory<int>)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (sycl::dynamic_work_group_memory<int>)' lvalue Function {{.*}} 'ff_8' 'void (sycl::dynamic_work_group_memory<int>)'
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::dynamic_work_group_memory<int>' Var {{.*}} 'DynMem' 'sycl::dynamic_work_group_memory<int>'

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_9(sycl::accessor<int, 1, sycl::access::mode::read_write> acc) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::accessor<int, 1, sycl::access::mode::read_write>)'
// CHECK-NEXT: ParmVarDecl {{.*}}acc 'sycl::accessor<int, 1, sycl::access::mode::read_write>'
// CHECK: ParmVarDecl {{.*}}__arg_Ptr '__global int *'
// CHECK: ParmVarDecl {{.*}}__arg_AccessRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_MemRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_Offset 'sycl::id<1>'
// CHECK: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}}.__init

template <typename DataT>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_9(sycl::local_accessor<DataT, 1> lacc) {
}

template void ff_9(sycl::local_accessor<float, 1> lacc);

// CHECK: FunctionDecl {{.*}}'void (sycl::local_accessor<float, 1>)'
// CHECK: ParmVarDecl {{.*}}lacc 'sycl::local_accessor<float, 1>'
// CHECK: ParmVarDecl {{.*}}__arg_Ptr '__local float *'
// CHECK: ParmVarDecl {{.*}}__arg_AccessRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_MemRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_Offset 'sycl::id<1>'
// CHECK: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}}.__init

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_9(sycl::local_accessor<int, 1> lacc) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::local_accessor<int, 1>)'
// CHECK: ParmVarDecl {{.*}}lacc 'sycl::local_accessor<int, 1>'
// CHECK: ParmVarDecl {{.*}}__arg_Ptr '__local int *'
// CHECK: ParmVarDecl {{.*}}__arg_AccessRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_MemRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}used __arg_Offset 'sycl::id<1>'
// CHECK: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}}.__init


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_10(sycl::sampler S) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::sampler)'
// CHECK: ParmVarDecl {{.*}}S 'sycl::sampler'
// CHECK: FunctionDecl {{.*}}'void (sampler_t) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}}__arg_Sampler 'sampler_t'
// CHECK: CXXMemberCallExpr {{.*}}'void'
// CHECK-NEXT: MemberExpr {{.*}}.__init


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_11(sycl::stream str) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::stream)'
// CHECK: ParmVarDecl {{.*}}str 'sycl::stream'
// CHECK: FunctionDecl {{.*}}'void (__global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>, int) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}}__arg_Ptr '__global char *'
// CHECK: ParmVarDecl {{.*}}__arg_AccessRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_MemRange 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}}__arg_Offset 'sycl::id<1>'
// CHECK: ParmVarDecl {{.*}}__arg__FlushBufferSize 'int'
// CHECK: CXXMemberCallExpr {{.*}}
// CHECK-NEXT: MemberExpr {{.*}}.__init


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_12(sycl::ext::oneapi::experimental::annotated_arg<int> arg) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::ext::oneapi::experimental::annotated_arg<int>)'
// CHECK: ParmVarDecl {{.*}}arg 'sycl::ext::oneapi::experimental::annotated_arg<int>'
// CHECK: FunctionDecl {{.*}}'void (int) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}}__arg__obj 'int'
// CHECK: CXXMemberCallExpr {{.*}}
// CHECK-NEXT: MemberExpr {{.*}}.__init


__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_13(sycl::ext::oneapi::experimental::annotated_ptr<int> ptr) {
}

// CHECK: FunctionDecl {{.*}}'void (sycl::ext::oneapi::experimental::annotated_ptr<int>)'
// CHECK: ParmVarDecl {{.*}}ptr 'sycl::ext::oneapi::experimental::annotated_ptr<int>'
// CHECK: FunctionDecl {{.*}}'void (int *) __attribute__((device_kernel))'
// CHECK: ParmVarDecl {{.*}}__arg__obj 'int *'
// CHECK: CXXMemberCallExpr {{.*}}
// CHECK-NEXT: MemberExpr {{.*}}.__init

