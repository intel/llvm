// RUN: %clangxx -target spir64 -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to fpga_mem at different scopes

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: %[[fpga_mem:.*fpga_mem.*]] = type { %[[fpga_mem_base:.*fpga_mem_base.*]] }
// CHECK: %[[fpga_mem_base]] = type { [2 x i32] }

constexpr intel::fpga_mem<int[2], decltype(oneapi::properties(intel::num_banks<4>))> global {9, 14};
// CHECK: @{{.*}}global = internal addrspace(1) constant { [2 x i32] } { [2 x i32] [i32 9, i32 14] }, align 4, !spirv.Decorations ![[GlobalProps:[0-9]+]]

// CHECK: @[[str:.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5827:\222\22}\00"

struct foo {
  int f;
  float h;
};

int main() {
  queue Q;
  int f = 0;
  foo b {2, 5.4f};

  Q.single_task([=]() {
    constexpr intel::fpga_mem<int[2], decltype(oneapi::properties(intel::num_banks<2>))> kernel {7, -1298};
  // CHECK: %[[kernel:kernel.*]] = alloca %[[fpga_mem]], align 8
  // CHECK: %[[kernel_acast:.*]] = addrspacecast ptr %[[kernel]] to ptr addrspace(4)
  // CHECK: %[[kernel_annot:.*]] = call dereferenceable(8) ptr addrspace(4) @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %[[kernel_acast]], ptr addrspace(1) @[[str]]
  // CHECK: getelementptr inbounds [2 x i32], ptr addrspace(4) %[[kernel_annot]]
    
    // Validate values of fpga_mem at various scopes
    static_assert(global[0] == 9);
    static_assert(global[1] == 14);
    static_assert(kernel[0] == 7);
    static_assert(kernel[1] == -1298);

    volatile int ReadVal = global[f] + kernel[f] + b.f;
  });
  return 0;
}

// CHECK: ![[GlobalProps]] = !{![[GlobalNumBanks:[0-9]+]], ![[GlobalResource:[0-9]+]]}
// CHECK-DAG: ![[GlobalNumBanks]] = !{i32 5827, i32 4}
// CHECK-DAG: ![[GlobalResource]] = !{i32 5826, [8 x i8] c"DEFAULT\00"}
