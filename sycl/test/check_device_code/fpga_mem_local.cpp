// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to each fpga_mem.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental;   // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: [[MemoryINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826:\22DEFAULT\22}\00"
// CHECK: [[ForcePow2DepthINTEL_FALSE:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5836:\220\22}\00"
// CHECK: [[ForcePow2DepthINTEL_TRUE:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5836:\221\22}\00"
// CHECK: [[DoublepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5831}\00"
// CHECK: [[SinglepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5830}\00"
// CHECK: [[MemoryINTEL_mlab:@.*]] = private unnamed_addr addrspace(1) constant [30 x i8] c"{5826:\22DEFAULT\22}{5826:\22MLAB\22}\00"
// CHECK: [[SimpleDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5833}\00"
// CHECK: [[TrueDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5885}\00"
// CHECK: [[MemoryINTEL_block_ram:@.*]] = private unnamed_addr addrspace(1) constant [35 x i8] c"{5826:\22DEFAULT\22}{5826:\22BLOCK_RAM\22}\00"
// CHECK: [[NumbanksINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5827:\224\22}\00"
// CHECK: [[StridesizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5883:\222\22}\00"
// CHECK: [[WordsizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5884:\228\22}\00"
// CHECK: [[MaxPrivateCopiesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5829:\223\22}\00"
// CHECK: [[MaxReplicatesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5832:\225\22}\00"

SYCL_EXTERNAL void fpga_mem_local() {
  int f = 5;
  intel::fpga_mem<int[10]> empty;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10],
                  decltype(oneapi::properties(intel::ram_stitching_min_ram))>
      min_ram;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_FALSE]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10],
                  decltype(oneapi::properties(intel::ram_stitching_max_fmax))>
      max_fmax;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_TRUE]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::clock_2x_true))>
      double_pumped;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[DoublepumpINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::clock_2x_false))>
      single_pumped;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[SinglepumpINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::resource_mlab))>
      mlab;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL_mlab]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(
                               intel::bi_directional_ports_false))>
      simple_dual_port;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[SimpleDualPortINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(
                               intel::bi_directional_ports_true))>
      true_dual_port;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[TrueDualPortINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10],
                  decltype(oneapi::properties(intel::resource_block_ram))>
      block_ram;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL_block_ram]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::num_banks<4>))>
      banks;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[NumbanksINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::stride_size<2>))>
      stride;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[StridesizeINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10], decltype(oneapi::properties(intel::word_size<8>))>
      word;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[WordsizeINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10],
                  decltype(oneapi::properties(intel::max_private_copies<3>))>
      copies;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MaxPrivateCopiesINTEL]]
  // CHECK-NOT: call void @llvm.memset
  intel::fpga_mem<int[10],
                  decltype(oneapi::properties(intel::num_replicates<5>))>
      replicates;
  // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MaxReplicatesINTEL]]
  // CHECK-NOT: call void @llvm.memset

  volatile int ReadVal =
      empty[f] + min_ram[f] + max_fmax[f] + double_pumped[f] +
      single_pumped[f] + mlab[f] + simple_dual_port[f] + true_dual_port[f] +
      block_ram[f] + banks[f] + stride[f] + word[f] + copies[f] + replicates[f];
}
