// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s
// This test checks if the parallel_for_work_item called indirecly from
// parallel_for_work_group gets the work_item_scope marker on it.
#include <sycl.hpp>

void foo(sycl::group<1> work_group) {
  work_group.parallel_for_work_item();
}

int main(int argc, char **argv) {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for_work_group(
         sycl::range<1>{1}, sycl::range<1>{1024}, ([=](sycl::group<1> wGroup) {
           foo(wGroup);
         }));
   });
  return 0;
}

// CHECK: define {{.*}} void @{{.*}}sycl{{.*}}group{{.*}}parallel_for_work_item{{.*}}(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %this) {{.*}}!work_item_scope {{.*}}!parallel_for_work_item
