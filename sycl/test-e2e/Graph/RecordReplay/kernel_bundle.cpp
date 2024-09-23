// RUN: %{build} -o %t.out
// RUN: %if cuda %{ %{run} %t.out %}
// RUN: %if level_zero %{env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --implicit-check-not=LEAK %}

// Checks the UR call trace to ensure that the bundle kernel of the single task
// is used.

// CHECK:---> urProgramCreateWithIL(
// CHECK-SAME: .phProgram = {{.*}} ([[PROGRAM_HANDLE1:[0-9a-fA-Fx]+]])
// CHECK-SAME: -> UR_RESULT_SUCCESS;
//
// CHECK:---> urProgramBuildExp(
// CHECK-SAME: .hProgram = [[PROGRAM_HANDLE1]]
//
// CHECK:---> urProgramRetain(
// CHECK-SAME: .hProgram = [[PROGRAM_HANDLE1]]
// CHECK-SAME: -> UR_RESULT_SUCCESS;

// CHECK:---> urKernelCreate(
// CHECK-SAME: .hProgram = [[PROGRAM_HANDLE1]]
// CHECK-SAME: .pKernelName = {{[0-9a-fA-Fx]+}} (_ZTS11Kernel1Name)
// CHECK-SAME: .phKernel = {{[0-9a-fA-Fx]+}}  ([[KERNEL_HANDLE:[0-9a-fA-Fx]+]])
// CHECK-SAME: -> UR_RESULT_SUCCESS;
//
// CHECK:---> urKernelRetain(
// CHECK-SAME: .hKernel = [[KERNEL_HANDLE]]
// CHECK-SAME: -> UR_RESULT_SUCCESS;
//
// CHECK:---> urCommandBufferAppendKernelLaunchExp(
// CHECK-SAME: .hKernel = [[KERNEL_HANDLE]]
//
// CHECK:---> urKernelRelease(
// CHECK-SAME: .hKernel = [[KERNEL_HANDLE]]
// CHECK-SAME: -> UR_RESULT_SUCCESS;

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/kernel_bundle.cpp"
