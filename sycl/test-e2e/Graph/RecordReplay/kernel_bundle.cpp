// RUN: %{build} -o %t.out
// RUN: %if cuda %{ %{run} %t.out %}
// RUN: %if level_zero %{env SYCL_PI_TRACE=2 %{run} %t.out | FileCheck %s --implicit-check-not=LEAK %}

// Checks the PI call trace to ensure that the bundle kernel of the single task
// is used.

// CHECK:---> piProgramCreate
// CHECK-NEXT: <unknown> : {{.*}}
// CHECK-NEXT: <unknown> : {{.*}}
// CHECK-NEXT: <unknown> : {{.*}}
// CHECK-NEXT: <unknown> : {{.*}}
// CHECK-NEXT: ) ---> pi_result : PI_SUCCESS
// CHECK-NEXT: [out]<unknown> ** : {{.*}}[ [[PROGRAM_HANDLE1:[0-9a-fA-Fx]]]
//
// CHECK:---> piProgramBuild(
// CHECK-NEXT: <unknown> : [[PROGRAM_HANDLE1]]
//
// CHECK:---> piProgramRetain(
// CHECK-NEXT: <unknown> : [[PROGRAM_HANDLE1]]
// CHECK-NEXT:---> pi_result : PI_SUCCESS

// CHECK:---> piKernelCreate(
// CHECK-NEXT: <unknown> : [[PROGRAM_HANDLE1]]
// CHECK-NEXT:<const char *>: _ZTS11Kernel1Name
// CHECK-NEXT: <unknown> : {{.*}}
// CHECK-NEXT: ---> pi_result : PI_SUCCESS
// CHECK-NEXT: [out]<unknown> ** : {{.*}}[ [[KERNEL_HANDLE:[0-9a-fA-Fx]]]
//
// CHECK:---> piKernelRetain(
// CHECK-NEXT: <unknown> : [[KERNEL_HANDLE]]
// CHECK-NEXT:---> pi_result : PI_SUCCESS
//
// CHECK:---> piextCommandBufferNDRangeKernel(
// CHECK-NEXT:<unknown> : {{.*}}
// CHECK-NEXT:<unknown> : [[KERNEL_HANDLE]]
//
// CHECK:---> piKernelRelease(
// CHECK-NEXT: <unknown> : [[KERNEL_HANDLE]]
// CHECK-NEXT:---> pi_result : PI_SUCCESS

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/kernel_bundle.cpp"
