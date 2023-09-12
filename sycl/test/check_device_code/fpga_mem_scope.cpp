// RUN: %clangxx -target spir64 -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to fpga_mem at different scopes

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// %[[fpga_mem:.*fpga_mem.*]] = type { %[[fpga_mem_base:.*fpga_mem_base.*]] }
// %[[fpga_mem_base]] = type { [2 x i32] }

constexpr intel::fpga_mem<int[2], decltype(oneapi::properties(oneapi::num_banks<4>))> global {9, 14};
// CHECK: @{{.*}}global = internal addrspace(1) constant { [2 x i32] } { [2 x i32] [i32 9, i32 14] }, align 4, !spirv.Decorations ![[GlobalProps:[0-9]+]]

struct foo {
  int f;
  float h;
};

int main() {
  queue Q;
  int f = 0;
  foo b {2, 5.4f};

  constexpr intel::fpga_mem<int[2], decltype(oneapi::properties(oneapi::num_banks<8>))> host {3, -9};

  Q.single_task([=]() {
    constexpr intel::fpga_mem<int[2], decltype(oneapi::properties(oneapi::num_banks<2>))> kernel {7, -1298};
  // %kernel.i = alloca %[[fpga_mem]], align 8
  // %kernel.ascast.i = addrspacecast ptr %kernel.i to ptr addrspace(4)
  // %6 = call dereferenceable(8) ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %kernel.ascast.i, ptr addrspace(1) @.str.12, ptr addrspace(1) @.str.1, i32 35, ptr addrspace(1) null)
  // %arrayidx9.i = getelementptr inbounds [2 x i32], ptr addrspace(4) %6, i64 0, i64 %idxprom.i
    
    // Validate values of fpga_mem at various scopes
    static_assert(global[0] == 9);
    static_assert(global[1] == 14);
    static_assert(host[0] == 3);
    static_assert(host[1] == -9);
    static_assert(kernel[0] == 7);
    static_assert(kernel[1] == -1298);


    volatile int ReadVal = global[f] + host[f] + kernel[f] + b.f;
  });
  return 0;
}

// CHECK: ![[GlobalProps]] = !{![[GlobalNumBanks:[0-9]+]], ![[GlobalResource:[0-9]+]]}
// CHECK-DAG: ![[GlobalNumBanks]] = !{i32 5827, i32 4}
// CHECK-DAG: ![[GlobalResource]] = !{i32 5826, [8 x i8] c"DEFAULT\00"}
