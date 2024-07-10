// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 FileCheck %s %} %else %{ %{run} %t.out %}

// REQUIRES: aspect-usm_shared_allocations

// Inteded - prefetch command not supported for OpenCL
// UNSUPPORTED: opencl

// Since Prefetch is only a memory hint that doesn't
// impact results but only performances, we verify
// that a node is correctly added by checking PI function calls

// CHECK: piextCommandBufferPrefetchUSM(
// CHECK-NEXT: <unknown> : 0x[[#%x,COMMAND_BUFFER:]]
// CHECK-NEXT: <unknown> : 0x[[#%x,PTR:]]
// CHECK-NEXT: <unknown> : 400
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 0x[[#%x,PREFETCH_SYNC_POINT:]]
// CHECK: pi_result : PI_SUCCESS

// CHECK: piextCommandBufferNDRangeKernel(
// CHECK-NEXT: <unknown> : 0x[[#COMMAND_BUFFER]]
// CHECK-NEXT: <unknown> : 0x[[#%x,KERNEL:]]
// CHECK-NEXT: <unknown> : 1
// CHECK-NEXT: <unknown> : 0x[[#%x,GLOBAL_WORK_OFFSET:]]
// CHECK-NEXT: <unknown> : 0x[[#%x,GLOBAL_WORK_SIZE:]]
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 1
// CHECK-NEXT: <unknown> : 0x[[#%x,SYNC_POINT_WAIT_LIST:]]
// CHECK-NEXT: <unknown> : 0x[[#%x,KERNEL_SYNC_POINT:]]
// CHECK: pi_result : PI_SUCCESS

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/prefetch.cpp"
