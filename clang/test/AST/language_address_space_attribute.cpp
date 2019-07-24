// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

// Verify that the language address space attribute is
// understood correctly by clang.

void langas() {
  // CHECK: VarDecl {{.*}} x_global '__global int *'
  __attribute__((ocl_global)) int *x_global;

  // CHECK: VarDecl {{.*}} z_global '__global int *'
  [[clang::ocl_global]] int *z_global;

  // CHECK: VarDecl {{.*}} x_local '__local int *'
  __attribute__((ocl_local)) int *x_local;

  // CHECK: VarDecl {{.*}} z_local '__local int *'
  [[clang::ocl_local]] int *z_local;

  // CHECK: VarDecl {{.*}} x_constant '__constant int *'
  __attribute__((ocl_constant)) int *x_constant;

  // CHECK: VarDecl {{.*}} z_constant '__constant int *'
  [[clang::ocl_constant]] int *z_constant;

  // CHECK: VarDecl {{.*}} x_private 'int *'
  __attribute__((ocl_private)) int *x_private;

  // CHECK: VarDecl {{.*}} z_private 'int *'
  [[clang::ocl_private]] int *z_private;

  // CHECK: VarDecl {{.*}} x_generic '__generic int *'
  __attribute__((ocl_generic)) int *x_generic;

  // CHECK: VarDecl {{.*}} z_generic '__generic int *'
  [[clang::ocl_generic]] int *z_generic;
}
