// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s

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
// CHECK: FunctionDecl {{.*}}test_kernel_handler{{.*}} 'void (int, char *)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} used  specialization_constants_buffer 'char *'

// Check declaration and initialization of kernel object local clone
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} '_arg_' 'int'

// Check declaration and initialization of kernel object local clone using default constructor
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'cl::sycl::kernel_handler':'cl::sycl::kernel_handler' 'void () noexcept'

// Check call to __init_specialization_constants_buffer
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (char *)' lvalue .__init_specialization_constants_buffer
