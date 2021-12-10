// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-unknown-unknown -ast-dump %s | FileCheck %s --check-prefix=NONATIVESUPPORT
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -ast-dump %s | FileCheck %s --check-prefix=NATIVESUPPORT

// This test checks that the compiler handles kernel_handler type (for
// SYCL 2020 specialization constants) correctly.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

int main() {
  q.submit([&](handler &h) {
    int a;
    kernel_handler kh;

    h.single_task<class test_kernel_handler>(
        [=](auto) {
          int local = a;
        },
        kh);
    h.parallel_for_work_group<class test_pfwg_kernel_handler>(
        [=](group<1> G, kernel_handler kh) {
          int local = a;
        },
        kh);
  });
}

// Check test_kernel_handler parameters
// NONATIVESUPPORT: FunctionDecl {{.*}}test_kernel_handler{{.*}} 'void (int, __global char *)'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used  _arg__specialization_constants_buffer '__global char *'

// Check declaration and initialization of kernel object local clone
// NONATIVESUPPORT-NEXT: CompoundStmt
// NONATIVESUPPORT-NEXT: DeclStmt
// NONATIVESUPPORT-NEXT: VarDecl {{.*}} cinit
// NONATIVESUPPORT-NEXT: InitListExpr
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check declaration and initialization of kernel handler local clone using default constructor
// NONATIVESUPPORT-NEXT: DeclStmt
// NONATIVESUPPORT-NEXT: VarDecl {{.*}} callinit
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler' 'void () noexcept'

// Check call to __init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: CXXMemberCallExpr {{.*}} 'void'
// NONATIVESUPPORT-NEXT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'sycl::kernel_handler' lvalue Var {{.*}} 'kh'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'char *' <AddressSpaceConversion>
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg__specialization_constants_buffer' '__global char *'
// NONATIVESUPPORT-NEXT: CompoundStmt
// NONATIVESUPPORT-NEXT: CXXOperatorCallExpr
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'void (*)(sycl::kernel_handler) const' <FunctionToPointerDecay>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'void (sycl::kernel_handler) const' lvalue CXXMethod {{.*}} 'operator()' 'void (sycl::kernel_handler) const'
// Kernel body with clones
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})'
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler':'sycl::kernel_handler' 'void (const sycl::kernel_handler &) noexcept'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const sycl::kernel_handler' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'sycl::kernel_handler' lvalue Var {{.*}} 'kh' 'sycl::kernel_handler'

// Check test_pfwg_kernel_handler parameters
// NONATIVESUPPORT: FunctionDecl {{.*}}test_pfwg_kernel_handler{{.*}} 'void (int, __global char *)'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used  _arg__specialization_constants_buffer '__global char *'

// Check declaration and initialization of kernel object local clone
// NONATIVESUPPORT-NEXT: CompoundStmt
// NONATIVESUPPORT-NEXT: DeclStmt
// NONATIVESUPPORT-NEXT: VarDecl {{.*}} cinit
// NONATIVESUPPORT-NEXT: InitListExpr
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check declaration and initialization of kernel handler local clone using default constructor
// NONATIVESUPPORT-NEXT: DeclStmt
// NONATIVESUPPORT-NEXT: VarDecl {{.*}} callinit
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler' 'void () noexcept'

// Check call to __init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: CXXMemberCallExpr {{.*}} 'void'
// NONATIVESUPPORT-NEXT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'sycl::kernel_handler' lvalue Var {{.*}} 'kh'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'char *' <AddressSpaceConversion>
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg__specialization_constants_buffer' '__global char *'
// NONATIVESUPPORT-NEXT: CompoundStmt
// NONATIVESUPPORT-NEXT: ExprWithCleanups
// NONATIVESUPPORT-NEXT: CXXOperatorCallExpr
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'void (*)(group<1>, sycl::kernel_handler) const' <FunctionToPointerDecay>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'void (group<1>, sycl::kernel_handler) const' lvalue CXXMethod {{.*}} 'operator()' 'void (group<1>, sycl::kernel_handler) const'

// Kernel body with clones
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})'
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'group<1>':'sycl::group<1>' 'void (sycl::group<1> &&) noexcept'
// NONATIVESUPPORT-NEXT: MaterializeTemporaryExpr
// NONATIVESUPPORT-NEXT: CXXTemporaryObjectExpr
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler' 'void (const sycl::kernel_handler &) noexcept'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const sycl::kernel_handler' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'sycl::kernel_handler' lvalue Var {{.*}} 'kh' 'sycl::kernel_handler'

// Test AST for default SPIR architecture

// Check test_kernel_handler parameters
// NATIVESUPPORT: FunctionDecl {{.*}}test_kernel_handler{{.*}} 'void (int, __global char *)'
// NATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'
// NATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used  _arg__specialization_constants_buffer '__global char *'

// Check declaration and initialization of kernel object local clone
// NATIVESUPPORT-NEXT: CompoundStmt
// NATIVESUPPORT-NEXT: DeclStmt
// NATIVESUPPORT-NEXT: VarDecl {{.*}} cinit
// NATIVESUPPORT-NEXT: InitListExpr
// NATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// NATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check declaration and initialization of kernel handler local clone using default constructor
// NATIVESUPPORT-NEXT: DeclStmt
// NATIVESUPPORT-NEXT: VarDecl {{.*}} callinit
// NATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler' 'void () noexcept'

// Check no call to __init_specialization_constants_buffer
// NATIVESUPPORT-NOT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer

// Kernel body with clones
// NATIVESUPPORT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue
// NATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})'
// NATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'sycl::kernel_handler':'sycl::kernel_handler' 'void (const sycl::kernel_handler &) noexcept'
// NATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const sycl::kernel_handler' lvalue
// NATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'sycl::kernel_handler' lvalue Var {{.*}} 'kh' 'sycl::kernel_handler'
