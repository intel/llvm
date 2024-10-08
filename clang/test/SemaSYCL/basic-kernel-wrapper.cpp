// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include "sycl.hpp"

sycl::queue myQueue;

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write> readWriteAccessor;

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_wrapper>(
        [=]() {
          readWriteAccessor.use();
        });
  });
}

// Check declaration of the kernel

// CHECK: FunctionDecl {{.*}}kernel_wrapper{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'

// Check parameters of the kernel

// CHECK: ParmVarDecl {{.*}} used [[_arg_Mem:[0-9a-zA-Z_]+]] '__global int *'
// CHECK: ParmVarDecl {{.*}} used [[_arg_AccessRange:[0-9a-zA-Z_]+]] 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_MemRange:[0-9a-zA-Z_]+]] 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_Offset:[0-9a-zA-Z_]+]] 'sycl::id<1>'

// Check body of the kernel

// Check lambda declaration inside the wrapper

// CHECK: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used __SYCLKernel '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})'

// Check accessor initialization

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ({{.*}}PtrType, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::accessor<int, 1, sycl::access::mode::read_write>' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue Var

// CHECK-NEXT: ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '[[_arg_Mem]]' '__global int *'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const range<1>':'const sycl::range<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_AccessRange]]' 'sycl::range<1>'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const range<1>':'const sycl::range<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_MemRange]]' 'sycl::range<1>'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'id<1>':'sycl::id<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const id<1>':'const sycl::id<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::id<1>' lvalue ParmVar {{.*}} '[[_arg_Offset]]' 'sycl::id<1>'

// Check that body of the kernel caller function is included into kernel

// CHECK: CompoundStmt {{.*}}
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}basic-kernel-wrapper.cpp{{.*}})' lvalue Var

// Check kernel wrapper attributes

// CHECK: OpenCLKernelAttr {{.*}} Implicit
// CHECK: ArtificialAttr {{.*}} Implicit
// CHECK: AsmLabelAttr {{.*}} Implicit "{{.*}}kernel_wrapper{{.*}}"
