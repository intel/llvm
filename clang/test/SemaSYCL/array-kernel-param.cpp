// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// arrays, Accessor arrays, and structs containing Accessors.

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

template <typename T>
struct S {
  T a[3];
};

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  Accessor acc[2];
  int a[2];
  int *a_ptrs[2];

  struct struct_acc_t {
    Accessor member_acc[2];
  } struct_acc;
  S<int> s;

  struct foo_inner {
    int foo_inner_x;
    int foo_inner_y;
    int *foo_inner_z[2];
  };

  struct foo {
    int foo_a;
    foo_inner foo_b[2];
    int *foo_2D[2][1];
    int foo_c;
  };

  // Not decomposed.
  struct foo2 {
    int foo_a;
    int foo_2D[2][1];
    int foo_c;
  };

  foo struct_array[2];
  foo2 struct_array2[2];

  int array_2D[2][3];

  a_kernel<class kernel_A>(
      [=]() {
        acc[1].use();
      });

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[1];
      });

  a_kernel<class kernel_B_ptrs>(
      [=]() {
        int local = *a_ptrs[1];
      });

  a_kernel<class kernel_C>(
      [=]() {
        struct_acc.member_acc[2].use();
      });

  a_kernel<class kernel_D>(
      [=]() {
        foo local = struct_array[1];
      });

  a_kernel<class kernel_E>(
      [=]() {
        int local = s.a[2];
      });

  a_kernel<class kernel_F>(
      [=]() {
        int local = array_2D[1][1];
      });

  a_kernel<class kernel_G>(
      [=]() {
        foo2 local = struct_array2[0];
      });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (__global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::id<1>'
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__wrapper_class'
// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int [2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'

// Check kernel_B_ptrs parameters
// CHECK: FunctionDecl {{.*}}kernel_B_ptrs{{.*}} 'void (__global int *, __global int *)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// Check kernel_B_ptrs inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_' '__global int *'

// Check kernel_C parameters
// CHECK: FunctionDecl {{.*}}kernel_C{{.*}} 'void (__global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}} 'struct_acc_t'
// CHECK-NEXT: InitListExpr {{.*}} 'Accessor [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Accessor'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'Accessor'

// Check __init functions are called
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init

// Check kernel_D parameters
// CHECK: FunctionDecl {{.*}}kernel_D{{.*}} 'void (int, int, int, __wrapper_class, __wrapper_class, int, int, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, int, int, int, int, __wrapper_class, __wrapper_class, int, int, __wrapper_class, __wrapper_class, __wrapper_class, __wrapper_class, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_c 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_a 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_x 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_y 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_inner_z '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_2D '__wrapper_class'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_foo_c 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'

// Initializer for struct array i.e. foo struct_array[2]
// CHECK-NEXT: InitListExpr {{.*}} 'foo [2]'

// Initializer for first element of struct_array
// CHECK-NEXT: InitListExpr {{.*}} 'foo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_a' 'int'

// Initializer for struct array inside foo i.e. foo_inner foo_b[2]
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner [2]'
// Initializer for first element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// Initializer for second element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][1]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_2D' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_2D' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_c' 'int'

// Initializer for second element of struct_array
// CHECK-NEXT: InitListExpr {{.*}} 'foo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_a' 'int'

// Initializer for struct array inside foo i.e. foo_inner foo_b[2]
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner [2]'
// Initializer for first element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// Initializer for second element of inner struct array
// CHECK-NEXT: InitListExpr {{.*}} 'foo_inner'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_x' 'int'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_foo_inner_y' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_inner_z' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[2][1]'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_2D' '__wrapper_class'
// CHECK-NEXT: InitListExpr {{.*}} 'int *[1]'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global int *'
// CHECK-NEXT: MemberExpr {{.*}} '__global int *' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_foo_2D' '__wrapper_class'

// Check kernel_E parameters
// CHECK: FunctionDecl {{.*}}kernel_E{{.*}} 'void (S<int>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'S<int>':'S<int>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used '(lambda at {{.*}}array-kernel-param.cpp{{.*}})' cinit
// CHECK-NEXT: InitListExpr {{.*}} '(lambda at {{.*}}array-kernel-param.cpp{{.*}})'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'S<int>':'S<int>' 'void (const S<int> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'S<int>':'S<int>' lvalue ParmVar {{.*}} '_arg_' 'S<int>':'S<int>'

// Check kernel_F parameters
// CHECK: FunctionDecl {{.*}}kernel_F{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__wrapper_class'
// Check kernel_F inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int [2][3]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int [3]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [3]' lvalue
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int [3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [3]' lvalue
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int [3]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)[3]' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int [2][3]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'int [2][3]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned

// Check kernel_G parameters.
// CHECK: FunctionDecl {{.*}}kernel_G{{.*}} 'void (__wrapper_class)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__wrapper_class'
// Check kernel_G inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'foo2 [2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'foo2 [2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'foo2 [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'foo2' 'void (const foo2 &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const foo2' lvalue <NoOp>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'foo2' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'foo2 *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'foo2 [2]' lvalue
// CHECK-NEXT: MemberExpr {{.*}} 'foo2 [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '__wrapper_class' lvalue ParmVar {{.*}} '_arg_' '__wrapper_class'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned
