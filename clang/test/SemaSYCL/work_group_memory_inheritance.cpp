// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// Check that AST is correctly generated for kernel arguments that inherit from work group memory.

#include "sycl.hpp"

sycl::queue myQueue;

struct WorkGroupMemoryDerived :
                         sycl::work_group_memory<int> {
};

int main() {
  myQueue.submit([&](sycl::handler &h) {
    WorkGroupMemoryDerived DerivedObject{ h };
    h.parallel_for<class kernel>([=] {
          DerivedObject.use();
        });
  });
  return 0;
}

// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (__local int *)'
// CHECK-NEXT: ParmVarDecl {{.*}}used _arg__base '__local int *'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: VarDecl {{.*}} used __SYCLKernel {{.*}} cinit
// CHECK-NEXT: InitListExpr {{.*}}
// CHECK-NEXT: InitListExpr {{.*}} 'WorkGroupMemoryDerived'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::work_group_memory<int>' 'void () noexcept'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__local int *)' lvalue .__init {{.*}}
// CHECK-NEXT: MemberExpr {{.*}} 'WorkGroupMemoryDerived' lvalue .DerivedObject
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Var {{.*}} '__SYCLKernel'
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__local int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__local int *' lvalue ParmVar {{.*}} '_arg__base' '__local int *'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'auto (*)() const -> void' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}}'auto () const -> void' lvalue CXXMethod {{.*}} 'operator()' 'auto () const -> void'
// CHECK-NEXT: ImplicitCastExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}}lvalue Var {{.*}} '__SYCLKernel'

