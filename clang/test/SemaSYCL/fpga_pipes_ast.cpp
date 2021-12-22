// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

// Tests for AST of SYCLIntelPipeIOAttr.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

struct pipe_storage {};

// CHECK: VarDecl {{.*}} Storage1 'const pipe_storage' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'const pipe_storage' 'void () noexcept'
// CHECK-NEXT: SYCLIntelPipeIOAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 5
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
const pipe_storage Storage1 __attribute__((io_pipe_id(5)));

// CHECK: VarTemplateDecl {{.*}} Storage2
// CHECK: VarDecl {{.*}} Storage2 'pipe_storage' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'pipe_storage' 'void () noexcept'
// CHECK_NEXT: SYCLIntelPipeIOAttr
// CHECK_NEXT: DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'N' 'int'
// CHECK: FunctionDecl {{.*}} used foo 'void (pipe_storage)'
// CHECK: VarTemplateSpecializationDecl {{.*}} used Storage2 'pipe_storage' callinit
// CHECK-NEXT: TemplateArgument integral 2
// CHECK-NEXT: CXXConstructExpr {{.*}} 'pipe_storage' 'void () noexcept'
// CHECK-NEXT: SYCLIntelPipeIOAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 N
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
template <int N>
pipe_storage Storage2 __attribute__((io_pipe_id(N)));

void foo(pipe_storage PS) {}

int main() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() {});
  });
  foo(Storage2<2>);
}
