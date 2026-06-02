// REQUIRES: aspect-usm_device_allocations

// Check to ensure that the SYCL runtime sets the
// UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS property to true for SYCLBIN kernels.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable %{sycl_target_opts} %{syclbin_exec_opts} %S/Inputs/basic_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} env SYCL_UR_TRACE=-1 %t.out %t.syclbin | FileCheck %s

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/basic.hpp"

// CHECK: ---> urKernelSetExecInfo
// CHECK-NEXT: <--- urKernelSetExecInfo(.hKernel = 0x{{[0-9a-fA-F]+}}, .propName = UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS, .propSize = 1, .pProperties = nullptr, .pPropValue = 0x{{[0-9a-fA-F]+}} (true)) -> UR_RESULT_SUCCESS;
