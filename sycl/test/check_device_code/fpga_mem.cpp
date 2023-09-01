// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to each fpga_mem.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: [[MemoryINTEL:@.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"{5826}\00"
// CHECK: [[ForcePow2DepthINTEL_FALSE:@.*]] = private unnamed_addr addrspace(1) constant [21 x i8] c"{5826}{5836:\22false\22}\00"
// CHECK: [[ForcePow2DepthINTEL_TRUE:@.*]] = private unnamed_addr addrspace(1) constant [20 x i8] c"{5826}{5836:\22true\22}\00"
// CHECK: [[DoublepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{5826}{5831}\00"
// CHECK: [[SinglepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{5826}{5830}\00"
// CHECK: [[MemoryINTEL_mlab:@.*]] = private unnamed_addr addrspace(1) constant [20 x i8] c"{5826}{5826:\22mlab\22}\00"
// CHECK: [[SimpleDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{5826}{5833}\00"
// CHECK: [[TrueDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [13 x i8] c"{5826}{5885}\00"
// CHECK: [[MemoryINTEL_block_ram:@.*]] = private unnamed_addr addrspace(1) constant [25 x i8] c"{5826}{5826:\22block_ram\22}\00"
// CHECK: [[NumbanksINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826}{5827:\224\22}\00"
// CHECK: [[StridesizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826}{5883:\222\22}\00"
// CHECK: [[WordsizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826}{5884:\228\22}\00"
// CHECK: [[MaxPrivateCopiesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826}{5829:\223\22}\00"
// CHECK: [[MaxReplicatesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826}{5832:\225\22}\00"

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    intel::fpga_mem<int[10]> empty; 
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %empty{{.*}}, ptr addrspace(1) [[MemoryINTEL]]
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
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::bi_directional_ports_false))> simple_dual_port;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %simple_dual_port{{.*}}, ptr addrspace(1) [[SimpleDualPortINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::bi_directional_ports_true))> true_dual_port;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %true_dual_port{{.*}}, ptr addrspace(1) [[TrueDualPortINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::resource_block_ram))> block_ram;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %block_ram{{.*}}, ptr addrspace(1) [[MemoryINTEL_block_ram]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::num_banks<4>))> banks;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %banks{{.*}}, ptr addrspace(1) [[NumbanksINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::stride_size<2>))> stride;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %stride{{.*}}, ptr addrspace(1) [[StridesizeINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::word_size<8>))> word;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %word{{.*}}, ptr addrspace(1) [[WordsizeINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::max_private_copies<3>))> copies;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %copies{{.*}}, ptr addrspace(1) [[MaxPrivateCopiesINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::num_replicates<5>))> replicates;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) %replicates{{.*}}, ptr addrspace(1) [[MaxReplicatesINTEL]]

    volatile int ReadVal = word[f];
  });
  return 0;
}


// CHECK: ...
