// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -std=c++20 -fsyntax-only %s -verify=device -ast-dump | FileCheck %s
// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -std=c++20 -fsyntax-only %s -verify=host

// This test checks that when a binding declaration is captured that
// we don't dereference the null VarDecl.  Also checks that the kernel
// parameter has the name of the binding declaration associated with it.

#include "sycl.hpp"

// host-no-diagnostics
// device-no-diagnostics

void foo() {
  int a[2] = {1, 2};
  auto [bind_x, bind_y] = a;
  auto Lambda = [=]() { bind_x = 10; };
  sycl::handler h;
  h.single_task<class C>(Lambda);
}

// CHECK: FunctionDecl {{.*}}foo{{.*}} 'void (int)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_bind_x 'int'
