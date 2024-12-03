// RUN: %{build} -o %t.out
// RUN: %if cuda %{ %{run} %t.out %}
// RUN: %if level_zero %{env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --implicit-check-not=LEAK %}

// Checks the UR call trace to ensure that the bundle kernel of the single task
// is used.

// CHECK:<--- urProgramCreateWithIL(
// CHECK-SAME: .phProgram = {{.*}} ([[PROGRAM_HANDLE1:[0-9a-fA-Fx]+]])

//
// CHECK:<--- urProgramBuildExp(
// CHECK-SAME: .hProgram = [[PROGRAM_HANDLE1]]
//
// CHECK:<--- urProgramRetain(.hProgram = [[PROGRAM_HANDLE1]]) -> UR_RESULT_SUCCESS

// CHECK:<--- urKernelCreate(
// CHECK-SAME: .hProgram = [[PROGRAM_HANDLE1]]
// CHECK-SAME: .pKernelName = {{.*}} (_ZTS11Kernel1Name)
// CHECK-SAME: .phKernel = {{.*}} ([[KERNEL_HANDLE:[0-9a-fA-Fx]+]])
// CHECK-SAME: -> UR_RESULT_SUCCESS
//
// CHECK:<--- urKernelRetain(.hKernel = [[KERNEL_HANDLE]]) -> UR_RESULT_SUCCESS
//
// CHECK:<--- urCommandBufferAppendKernelLaunchExp(
// CHECK-SAME: .hKernel = [[KERNEL_HANDLE]]
//
// CHECK:<--- urKernelRelease(.hKernel = [[KERNEL_HANDLE]]) -> UR_RESULT_SUCCESS

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/kernel_bundle.cpp"
