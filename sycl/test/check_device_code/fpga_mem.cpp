// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to each fpga_mem.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: [[ForcePow2DepthINTEL_FALSE:@.*]] = private unnamed_addr addrspace(1) constant [15 x i8] c"{5836:\22false\22}\00"
// CHECK: [[ForcePow2DepthINTEL_TRUE:@.*]] = private unnamed_addr addrspace(1) constant [14 x i8] c"{5836:\22true\22}\00"
// CHECK: [[DoublepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [14 x i8] c"{5831:\22true\22}\00"
// CHECK: [[SinglepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [14 x i8] c"{5830:\22true\22}\00"
// CHECK: [[MemoryINTEL_mlab:@.*]] = private unnamed_addr addrspace(1) constant [14 x i8] c"{5826:\22mlab\22}\00"

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    // [[intel::num_banks(888)]]int a [10];
    intel::fpga_mem<int[10]> empty; //TODO artem:TEST
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::ram_stitching_min_ram))> min_ram;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %min_ram{{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_FALSE]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::ram_stitching_max_fmax))> max_fmax;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %max_fmax{{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_TRUE]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::clock_2x_true))> double_pumped;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %double_pumped{{.*}}, ptr addrspace(1) [[DoublepumpINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::clock_2x_true))> single_pumped;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %single_pumped{{.*}}, ptr addrspace(1) [[SinglepumpINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::resource_mlab))> mlab;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %mlab{{.*}}, ptr addrspace(1) [[MemoryINTEL_mlab]]

    volatile int ReadVal = empty[f];
  });
  return 0;
}


// CHECK: ...
