// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s

// Check that when parallel_for_work_item is reached from the
// parallel_for_work_group lambda through several levels of calls, every
// function on the call path gets the work_group_scope marker (so that the
// LowerWGScope pass guards its work-group scope code), while functions that do
// not lead to a parallel_for_work_item are left unmarked.

#include <sycl.hpp>

void leaf(sycl::group<1> G) { G.parallel_for_work_item(); }

void mid(sycl::group<1> G) { leaf(G); }

void top(sycl::group<1> G) { mid(G); }

void plain(int *P) { ++*P; }

int main() {
  sycl::queue Q;
  int *P = nullptr;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group(
        sycl::range<1>{1}, sycl::range<1>{16}, [=](sycl::group<1> G) {
          plain(P);
          top(G);
        });
  });
  return 0;
}

// CHECK-DAG: define {{.*}}spir_func void @_Z3topN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z3midN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @_Z4leafN4sycl3_V15groupILi1EEE({{.*}}!work_group_scope
// CHECK-DAG: define {{.*}}spir_func void @{{.*}}group{{.*}}parallel_for_work_item{{.*}}!work_item_scope {{.*}}!parallel_for_work_item
// A helper which does not reach parallel_for_work_item is not marked:
// CHECK-DAG: define {{.*}}spir_func void @_Z5plainPi({{.*}}) #{{[0-9]+}} {{[{]$}}
