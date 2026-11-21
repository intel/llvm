// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s

// Check that when parallel_for_work_item is reached from the
// parallel_for_work_group lambda through several levels of calls, every
// function on the call path gets the work_group_scope marker (so that the
// LowerWGScope pass guards its work-group scope code), while functions that do
// not lead to a parallel_for_work_item are left unmarked.
//
// 'alternate' reaches parallel_for_work_item through 'leaf', which is already
// marked by the time this second path is visited; the walk up the call stack
// must keep going past it so that 'alternate' is marked as well.
//
// The walk must also stop at the kernel: the generated kernel does not get a
// scope marker. The corresponding AST checks are in
// SemaSYCL/sycl-scope-attr-indirect-ast.cpp.

#include <sycl.hpp>

void leaf(sycl::group<1> G) { G.parallel_for_work_item(); }

void mid(sycl::group<1> G) { leaf(G); }

void top(sycl::group<1> G) { mid(G); }

void alternate(sycl::group<1> G) { leaf(G); }

void plain(int *P) { ++*P; }

int main() {
  sycl::queue Q;
  int *P = nullptr;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group(
        sycl::range<1>{1}, sycl::range<1>{16}, [=](sycl::group<1> G) {
          plain(P);
          top(G);
          alternate(G);
        });
  });
  return 0;
}

// The generated kernel is the first function emitted, so check it before the
// CHECK-DAG block; the -SAME keeps the -NOT on the kernel's define line.
// CHECK: define {{.*}}spir_kernel void @
// CHECK-NOT: !work_group_scope
// CHECK-SAME: {{[{]$}}

// CHECK-DAG: define {{.*}}spir_func void @_Z3topN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z3midN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z4leafN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z9alternateN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}group{{.*}}parallel_for_work_item{{.*}}!work_item_scope {{.*}}!parallel_for_work_item
// A helper which does not reach parallel_for_work_item is not marked:
// CHECK-DAG: define {{.*}}spir_func void @_Z5plainPi({{.*}}) #{{[0-9]+}} {{[{]$}}
