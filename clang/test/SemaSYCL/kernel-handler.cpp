// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-unknown-unknown -ast-dump %s | FileCheck %s --check-prefix=NONATIVESUPPORT
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -ast-dump %s | FileCheck %s --check-prefix=NATIVESUPPORT

// This test checks that the compiler handles kernel_handler type (for
// SYCL 2020 specialization constants) correctly.

//FIXME: Move to headers
namespace cl {
namespace sycl {
class kernel_handler {
  void __init_specialization_constants_buffer(char *specialization_constants_buffer) {}
};
} // namespace sycl
} // namespace cl

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc, cl::sycl::kernel_handler kh) {
  kernelFunc(kh);
}

int main() {
  int a;
  cl::sycl::kernel_handler kh;

  a_kernel<class test_kernel_handler>(
      [=](auto) {
        int local = a;
      },
      kh);
}

// Check test_kernel_handler parameters
// NONATIVESUPPORT: FunctionDecl {{.*}}test_kernel_handler{{.*}} 'void (int, char *)'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'
// NONATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used  specialization_constants_buffer 'char *'

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
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' 'void () noexcept'

// Check call to __init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: CXXMemberCallExpr {{.*}} 'void'
// NONATIVESUPPORT-NEXT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' lvalue Var {{.*}} 'kh'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'char *' <LValueToRValue>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'char *' lvalue ParmVar {{.*}} 'specialization_constants_buffer' 'char *'
// NONATIVESUPPORT-NEXT: CompoundStmt
// NONATIVESUPPORT-NEXT: CXXOperatorCallExpr
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'void (*)(cl::sycl::kernel_handler) const' <FunctionToPointerDecay>
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'void (cl::sycl::kernel_handler) const' lvalue CXXMethod {{.*}} 'operator()' 'void (cl::sycl::kernel_handler) const'
// Kernel body with clones
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})'
// NONATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' 'void (const cl::sycl::kernel_handler &) noexcept'
// NONATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const cl::sycl::kernel_handler' lvalue
// NONATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' lvalue Var {{.*}} 'kh' 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler'

// Check test_kernel_handler parameters
// NATIVESUPPORT: FunctionDecl {{.*}}test_kernel_handler{{.*}} 'void (int)'
// NATIVESUPPORT-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'

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
// NATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' 'void () noexcept'

// Check no call to __init_specialization_constants_buffer
// NATIVESUPPORT-NOT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer

// Kernel body with clones
// NATIVESUPPORT: ImplicitCastExpr {{.*}} 'const (lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue
// NATIVESUPPORT-NEXT: DeclRefExpr {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})' lvalue Var {{.*}} '(lambda at {{.*}}kernel-handler.cpp{{.*}})'
// NATIVESUPPORT-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' 'void (const cl::sycl::kernel_handler &) noexcept'
// NATIVESUPPORT-NEXT: ImplicitCastExpr {{.*}} 'const cl::sycl::kernel_handler' lvalue
// NATIVESUPPORT-NEXT: DeclRefExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' lvalue Var {{.*}} 'kh' 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler'
