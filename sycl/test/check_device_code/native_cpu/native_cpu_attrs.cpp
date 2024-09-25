// REQUIRES: native_cpu_ock
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-native-dump-device-ir %s | FileCheck %s

#include "sycl.hpp"
using namespace sycl;

class Test;
int main() {
  sycl::queue deviceQueue;
  sycl::nd_range<2> r({1, 1}, {1,1});
  deviceQueue.submit([&](handler &h) {
    h.parallel_for<Test>(
        r, [=](nd_item<2> it) { 
          it.barrier(access::fence_space::local_space);
          //CHECK-DAG: call void @__mux_work_group_barrier({{.*}})
          atomic_fence(memory_order::acquire, memory_scope::work_group);
          //CHECK-DAG: call void @__mux_mem_barrier({{.*}})
        });
  });

}

//CHECK-DAG: define{{.*}}@__mux_work_group_barrier{{.*}}#[[ATTR:[0-9]+]]
//CHECK-DAG: [[ATTR]]{{.*}}convergent

//CHECK-DAG: define{{.*}}@__mux_mem_barrier{{.*}}#[[ATTR_MEM:[0-9]+]]
//CHECK-DAG: [[ATTR_MEM]]{{.*}}convergent
