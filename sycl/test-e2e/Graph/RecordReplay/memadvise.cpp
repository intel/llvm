// RUN: %{build} -o %t.out
// RUN: %if linux && (level_zero || cuda) %{ env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 FileCheck %s %} %else %{ %{run} %t.out %}

// REQUIRES: aspect-usm_shared_allocations

// Intended - Mem advise command not supported for OpenCL
// UNSUPPORTED: opencl

// Since Mem advise is only a memory hint that doesn't
// impact results but only performances, we verify
// that a node is correctly added by checking UR function calls.

// CHECK: <--- urCommandBufferAppendUSMAdviseExp
// CHECK-SAME: .hCommandBuffer = 0x[[#%x,COMMAND_BUFFER:]]
// CHECK-SAME: .pMemory = 0x[[#%x,PTR:]]
// CHECK-SAME: .size = 400
// CHECK-SAME: .pSyncPoint = {{.*}} (0x[[#%x,ADVISE_SYNC_POINT:]])

// CHECK: <--- urCommandBufferAppendKernelLaunchExp(
// CHECK-SAME: .hCommandBuffer = 0x[[#%x,COMMAND_BUFFER]]
// CHECK-SAME: .hKernel = 0x[[#%x,KERNEL:]]
// CHECK-SAME: .workDim = 1
// CHECK-SAME: .pGlobalWorkOffset = 0x[[#%x,GLOBAL_WORK_OFFSET:]]
// CHECK-SAME: .pGlobalWorkSize = 0x[[#%x,GLOBAL_WORK_SIZE:]]
// CHECK-SAME: .pSyncPointWaitList = 0x[[#%x,SYNC_POINT_WAIT_LIST:]]
// CHECK-SAME: .pSyncPoint = 0x[[#%x,KERNEL_SYNC_POINT:]]
// CHECK-SAME: -> UR_RESULT_SUCCESS

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/memadvise.cpp"
