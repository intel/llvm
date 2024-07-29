// RUN: %{build} -o %t.out
// RUN: %if cuda %{ %{run} %t.out %}
// RUN: %if level_zero %{env SYCL_UR_TRACE=1 %{run} %t.out | FileCheck %s --implicit-check-not=LEAK %}

// TODO: Reenable, see https://github.com/intel/llvm/issues/14763
// UNSUPPORTED: windows, linux

// Checks the UR call trace to ensure that the bundle kernel of the single task
// is used.

// CHECK:---> urProgramCreate
// CHECK-SAME:, .phProgram = {{.*}} ([[PROGRAM_HANDLE1:[0-9a-fA-Fx]+]])
// CHECK-SAME: -> UR_RESULT_SUCCESS;
//
// CHECK:---> urProgramBuild(
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
