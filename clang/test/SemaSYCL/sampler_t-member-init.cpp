// RUN: %clang_cc1 %s -fsycl-is-device -ast-dump 2>&1 | FileCheck %s

const __ocl_sampler_t Global = 0;

class Foo {
  int i;
  __ocl_sampler_t Member;
  __ocl_sampler_t Member2;
  __ocl_sampler_t Member3;
  __ocl_sampler_t Member4;

  Foo(__ocl_sampler_t Param) :
    // CHECK: CXXConstructorDecl
    // CHECK-SAME: Foo 'void (__ocl_sampler_t)'
    Member(Param),
    // CHECK: CXXCtorInitializer Field
    // CHECK-SAME: 'Member' '__ocl_sampler_t':'sampler_t'
    // CHECK-NEXT: ImplicitCastExpr
    // CHECK-SAME: '__ocl_sampler_t':'sampler_t' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr
    // CHECK-SAME: lvalue ParmVar
    // CHECK-SAME: 'Param' '__ocl_sampler_t':'sampler_t'
    Member2(4),
    // CHECK: CXXCtorInitializer Field
    // CHECK-SAME: 'Member2' '__ocl_sampler_t':'sampler_t'
    // CHECK-NEXT: ImplicitCastExpr
    // CHECK-SAME: 'sampler_t' <IntToOCLSampler>
    // CHECK-NEXT: IntegerLiteral
    // CHECK-SAME: 'int' 4
    Member3(),
    // CHECK: CXXCtorInitializer Field
    // CHECK-SAME: 'Member3' '__ocl_sampler_t':'sampler_t'
    // CHECK-NEXT: ImplicitValueInitExpr
    // CHECK-SAME: '__ocl_sampler_t':'sampler_t'
    Member4(Global)
    // CHECK: CXXCtorInitializer Field
    // CHECK-SAME: 'Member4' '__ocl_sampler_t':'sampler_t'
    // CHECK-NEXT: ImplicitCastExpr
    // CHECK-SAME: '__ocl_sampler_t':'sampler_t' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr
    // CHECK-SAME: lvalue Var
    // CHECK-SAME: 'Global' 'const __ocl_sampler_t':'const sampler_t'
  {}
};

