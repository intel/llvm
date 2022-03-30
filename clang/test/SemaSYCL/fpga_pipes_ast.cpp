// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump %s | FileCheck %s

// Tests for AST of __attribute__((io_pipe_id(..))) attribute.

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

int main() {
  Storage2<2>;
  return 0;
}

// Test for Intel 'io_pipe_id' attribute duplication.
// No diagnostic is emitted because the arguments match.
// Duplicate attribute is silently ignored.
// CHECK: VarDecl {{.*}} Storage4 'const pipe_storage' callinit
// CHECK-NEXT: CXXConstructExpr {{.*}} 'const pipe_storage' 'void () noexcept'
// CHECK-NEXT: SYCLIntelPipeIOAttr
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NOT: SYCLIntelPipeIOAttr
const pipe_storage Storage4 __attribute__((io_pipe_id(1))) __attribute__((io_pipe_id(1)));
