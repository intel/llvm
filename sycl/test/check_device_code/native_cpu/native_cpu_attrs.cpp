// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-native-dump-device-ir %s &> %t.ll
// RUN: FileCheck %s --input-file %t.ll --check-prefix=CHECK-WG-BARRIER
// RUN: FileCheck %s --input-file %t.ll

#include <sycl/sycl.hpp>
using namespace sycl;

class Test;
int main() {
  sycl::queue deviceQueue;
  sycl::nd_range<2> r({1, 1}, {1,1});
  deviceQueue.submit([&](handler &h) {
    h.parallel_for<Test>(
        r, [=](nd_item<2> it) { 
          atomic_fence(memory_order::acquire, memory_scope::work_group);
          //CHECK-DAG: call void @__mux_mem_barrier({{.*}})
        });
  });

}

// Currently Native CPU uses the WorkItemLoops pass from the oneAPI
// Construction Kit to materialize barriers, so the builtin shouldn't 
// be referenced anymore in the module.
// CHECK-WG-BARRIER-NOT: @__mux_work_group_barrier

// CHECK-DAG: define{{.*}}@__mux_mem_barrier{{.*}}#[[ATTR_MEM:[0-9]+]]
// CHECK-DAG: [[ATTR_MEM]]{{.*}}convergent
