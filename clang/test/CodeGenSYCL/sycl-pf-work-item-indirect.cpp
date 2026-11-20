// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,KERNEL
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -internal-isystem %S/Inputs -ast-dump -ast-dump-filter kernel_parallel_for_work_group %s | FileCheck %s --check-prefix=AST

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
// The walk must also stop at the kernel: neither the generated kernel nor the
// SYCL kernel function (kernel_parallel_for_work_group in Inputs/sycl.hpp,
// which itself calls sycl::parallel_for_work_item directly) gets a scope
// marker.

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
// KERNEL: define {{.*}}spir_kernel void @
// KERNEL-NOT: !work_group_scope
// KERNEL-SAME: {{[{]$}}

// CHECK-DAG: define {{.*}}spir_func void @_Z3topN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z3midN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z4leafN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z9alternateN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}group{{.*}}parallel_for_work_item{{.*}}!work_item_scope {{.*}}!parallel_for_work_item
// A helper which does not reach parallel_for_work_item is not marked:
// CHECK-DAG: define {{.*}}spir_func void @_Z5plainPi({{.*}}) #{{[0-9]+}} {{[{]$}}

// The SYCL kernel function is not emitted in device IR, so check its AST
// instead. The instantiation is the last decl dumped, so the NOT runs to the
// end of its subtree, which is where an added attribute would appear.
// AST: FunctionDecl {{.*}} kernel_parallel_for_work_group {{.*}} implicit_instantiation
// AST-NOT: SYCLScopeAttr
