// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to each fpga_mem.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental;   // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// CHECK: [[MemoryINTEL:@.*]] = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826:\22DEFAULT\22}\00"
// CHECK: [[ForcePow2DepthINTEL_FALSE:@.*]] = private unnamed_addr addrspace(1) constant [31 x i8] c"{5826:\22DEFAULT\22}{5836:\22false\22}\00"
// CHECK: [[ForcePow2DepthINTEL_TRUE:@.*]] = private unnamed_addr addrspace(1) constant [30 x i8] c"{5826:\22DEFAULT\22}{5836:\22true\22}\00"
// CHECK: [[DoublepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5831}\00"
// CHECK: [[SinglepumpINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5830}\00"
// CHECK: [[MemoryINTEL_mlab:@.*]] = private unnamed_addr addrspace(1) constant [30 x i8] c"{5826:\22DEFAULT\22}{5826:\22mlab\22}\00"
// CHECK: [[SimpleDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5833}\00"
// CHECK: [[TrueDualPortINTEL:@.*]] = private unnamed_addr addrspace(1) constant [23 x i8] c"{5826:\22DEFAULT\22}{5885}\00"
// CHECK: [[MemoryINTEL_block_ram:@.*]] = private unnamed_addr addrspace(1) constant [35 x i8] c"{5826:\22DEFAULT\22}{5826:\22block_ram\22}\00"
// CHECK: [[NumbanksINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5827:\224\22}\00"
// CHECK: [[StridesizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5883:\222\22}\00"
// CHECK: [[WordsizeINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5884:\228\22}\00"
// CHECK: [[MaxPrivateCopiesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5829:\223\22}\00"
// CHECK: [[MaxReplicatesINTEL:@.*]] = private unnamed_addr addrspace(1) constant [27 x i8] c"{5826:\22DEFAULT\22}{5832:\225\22}\00"

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    intel::fpga_mem<int[10]> empty;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::ram_stitching_min_ram))>
        min_ram;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_FALSE]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::ram_stitching_max_fmax))>
        max_fmax;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[ForcePow2DepthINTEL_TRUE]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(intel::clock_2x_true))>
        double_pumped;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[DoublepumpINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::clock_2x_false))>
        single_pumped;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[SinglepumpINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(intel::resource_mlab))>
        mlab;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL_mlab]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(
                                 intel::bi_directional_ports_false))>
        simple_dual_port;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[SimpleDualPortINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(
                                 intel::bi_directional_ports_true))>
        true_dual_port;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[TrueDualPortINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::resource_block_ram))>
        block_ram;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MemoryINTEL_block_ram]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(intel::num_banks<4>))>
        banks;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[NumbanksINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::stride_size<2>))>
        stride;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[StridesizeINTEL]]
    intel::fpga_mem<int[10], decltype(oneapi::properties(intel::word_size<8>))>
        word;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[WordsizeINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::max_private_copies<3>))>
        copies;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MaxPrivateCopiesINTEL]]
    intel::fpga_mem<int[10],
                    decltype(oneapi::properties(intel::num_replicates<5>))>
        replicates;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[MaxReplicatesINTEL]]

    volatile int ReadVal = word[f];
  });
  return 0;
}
