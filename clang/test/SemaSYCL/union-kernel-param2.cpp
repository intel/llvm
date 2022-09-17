// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel arguments for
// a struct-with-an-array-of-unions and a array-of-struct-with-a-union.

#include "sycl.hpp"

sycl::queue myQueue;

int main() {

  union MyUnion {
    struct MyStruct {
      int a[3];
      float b;
      char c;
    } struct_mem;
    int d;
  } union_mem;

  struct MyStruct {
    union MyUnion {
      int a[3];
      float b;
      char c;
    } union_mem;
    sycl::accessor<char, 1, sycl::access::mode::read> AccField;
  } struct_mem;

  struct MyStructWithPtr {
    union MyUnion {
      int a[3];
      float b;
      char c;
    } union_mem;
    int *d;
  } structWithPtr_mem;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_A>(
        [=]() {
          int local = union_mem.struct_mem.a[2];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_B>(
        [=]() {
          int local = struct_mem.union_mem.a[2];
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_C>(
        [=]() {
          int local = structWithPtr_mem.union_mem.a[2];
        });
  });
}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (union MyUnion)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyUnion'

// Check kernel_A inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyUnion' 'void (const MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyUnion' lvalue ParmVar {{.*}} '_arg_union_mem' 'union MyUnion':'MyUnion'

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (union MyUnion, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_union_mem 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField '__global char *'
// CHECK: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField 'sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_AccField 'sycl::id<1>'

// Check kernel_B inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: InitListExpr {{.*}} 'MyStruct'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' 'void (const MyStruct::MyUnion &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyStruct::MyUnion'
// CHECK-NEXT: DeclRefExpr {{.*}} 'union MyUnion':'MyStruct::MyUnion' lvalue ParmVar {{.*}} '_arg_union_mem' 'union MyUnion':'MyStruct::MyUnion'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::accessor<char, 1, sycl::access::mode::read>'

// Check call to __init to initialize AccField
// CHECK-NEXT: CXXMemberCallExpr
// CHECK-NEXT: MemberExpr {{.*}} lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} lvalue .AccField
// CHECK-NEXT: MemberExpr {{.*}} lvalue .struct_mem
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}union-kernel-param2.cpp:48:9)' lvalue Var {{.*}} '__SYCLKernel' '(lambda at {{.*}}union-kernel-param2.cpp:48:9)'

// Check kernel_C parameters
// CHECK: FunctionDecl {{.*}}kernel_C{{.*}} 'void (_generated_MyStructWithPtr)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_structWithPtr_mem '_generated_MyStructWithPtr'

// Check kernel_C inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'struct MyStructWithPtr':'MyStructWithPtr' 'void () noexcept'

// Check call to __builtin_memcpy to initialize structWithPtr_mem
// CHECK-NEXT: CallExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *(*)(void *, const void *, unsigned long) noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} '__builtin_memcpy' 'void *(void *, const void *, unsigned long) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} 'struct MyStructWithPtr *' prefix '&' cannot overflow
// CHECK-NEXT: MemberExpr {{.*}} 'struct MyStructWithPtr':'MyStructWithPtr' lvalue .structWithPtr_mem
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}union-kernel-param2.cpp:55:9)' lvalue Var {{.*}} '__SYCLKernel' '(lambda at {{.*}}union-kernel-param2.cpp:55:9)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const void *' <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '_generated_MyStructWithPtr *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '_generated_MyStructWithPtr' lvalue ParmVar {{.*}} '_arg_structWithPtr_mem' '_generated_MyStructWithPtr'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 24
