// RUN: %clangxx -fsycl-device-only -S -emit-llvm -Xclang -disable-llvm-passes %s -o - | FileCheck %s

// Check that each work item stores its own private_memory address into the PFWI
// lambda object before the PFWI call. PM is the only capture, so it is at
// offset 0, and the checks do not depend on whether clang emits a GEP for that
// field.

#include <sycl/sycl.hpp>

using namespace sycl;

void test(queue &Q) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class K>(
        range<1>(1), range<1>(8), [=](group<1> G) {
          private_memory<int> PM(G);
          G.parallel_for_work_item(
              [&](h_item<1> I) { PM(I) = I.get_local_id(0); });
        });
  });
}

// CHECK-LABEL: define {{.*}} !work_group_scope
// CHECK:         %[[PM:[A-Za-z0-9_.]+]] = alloca %"class.sycl::_V1::private_memory"{{.*}}, !work_item_scope
// CHECK:         %[[OBJ:[A-Za-z0-9_.]+]] = alloca %class.anon
// CHECK:       LeaderMat:
// CHECK:         call void @llvm.memcpy.p0.p3.i64(ptr align {{[0-9]+}} %[[OBJ]], ptr addrspace(3)
// CHECK:         [[CAST:%[0-9]+]] = addrspacecast ptr %[[PM]] to ptr addrspace(4)
// CHECK-NEXT:    store ptr addrspace(4) [[CAST]], ptr addrspace(4) %{{.*}}
// CHECK-NEXT:    call spir_func void @{{.*}}parallel_for_work_item
