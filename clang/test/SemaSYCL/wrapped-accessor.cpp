// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper in case when
// accessor is wrapped.

#include "sycl.hpp"

sycl::queue myQueue;

template <typename Acc>
struct AccWrapper { Acc accessor; };

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write> acc;
  auto acc_wrapped = AccWrapper<decltype(acc)>{acc};

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class wrapped_access>(
        [=] {
          acc_wrapped.accessor.use();
        });
  });

  return 0;
}

// Check declaration of the kernel
// CHECK: wrapped_access{{.*}} 'void (__global int *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'

// Check parameters of the kernel
// CHECK: ParmVarDecl {{.*}} used _arg_accessor '__global int *'
// CHECK: ParmVarDecl {{.*}} used [[_arg_AccessRange:[0-9a-zA-Z_]+]] 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_MemRange:[0-9a-zA-Z_]+]] 'sycl::range<1>'
// CHECK: ParmVarDecl {{.*}} used [[_arg_Offset:[0-9a-zA-Z_]+]] 'sycl::id<1>'

// Check that wrapper object itself is initialized with corresponding kernel
// argument
// CHECK: VarDecl {{.*}}'(lambda at {{.*}}wrapped-accessor.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'(lambda at {{.*}}wrapped-accessor.cpp{{.*}})'
// CHECK-NEXT: InitListExpr {{.*}}'AccWrapper<sycl::accessor<int, 1, sycl::access::mode::read_write>>'
// CHECK-NEXT: CXXConstructExpr {{.*}}'sycl::accessor<int, 1, sycl::access::mode::read_write>' 'void () noexcept'

// Check that accessor field of the wrapper object is initialized using __init method
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ({{.*}}PtrType, range<1>, range<1>, id<1>)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::accessor<int, 1, sycl::access::mode::read_write>' lvalue .accessor {{.*}}
// CHECK-NEXT: MemberExpr {{.*}} 'AccWrapper<decltype(acc)>':'AccWrapper<sycl::accessor<int, 1, sycl::access::mode::read_write>>' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}wrapped-accessor.cpp{{.*}})'

// Parameters of the _init method
// CHECK-NEXT: ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global int *' lvalue ParmVar {{.*}} '_arg_accessor' '__global int *'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const range<1>':'const sycl::range<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_AccessRange]]' 'sycl::range<1>'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'range<1>':'sycl::range<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const range<1>':'const sycl::range<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::range<1>' lvalue ParmVar {{.*}} '[[_arg_MemRange]]' 'sycl::range<1>'

// CHECK-NEXT: CXXConstructExpr {{.*}} 'id<1>':'sycl::id<1>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const id<1>':'const sycl::id<1>' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::id<1>' lvalue ParmVar {{.*}} '[[_arg_Offset]]' 'sycl::id<1>'
