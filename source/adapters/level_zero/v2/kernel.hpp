//===--------- kernel.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../program.hpp"

#include "common.hpp"
#include "memory.hpp"

struct ur_single_device_kernel_t {
  ur_single_device_kernel_t(ur_device_handle_t hDevice,
                            ze_kernel_handle_t hKernel, bool ownZeHandle);
  ur_result_t release();

  ur_device_handle_t hDevice;
  v2::raii::ze_kernel_handle_t hKernel;
  mutable ZeCache<ZeStruct<ze_kernel_properties_t>> zeKernelProperties;
};

struct ur_kernel_handle_t_ : _ur_object {
private:
  struct pending_memory_allocation_t;

public:
  struct common_properties_t {
    std::string name;
    uint32_t numKernelArgs;
  };

  ur_kernel_handle_t_(ur_program_handle_t hProgram, const char *kernelName);

  // From native handle
  ur_kernel_handle_t_(ur_native_handle_t hNativeKernel,
                      ur_program_handle_t hProgram, ur_context_handle_t context,
                      const ur_kernel_native_properties_t *pProperties);

  // Get L0 kernel handle for a given device
  ze_kernel_handle_t getZeHandle(ur_device_handle_t hDevice);

  // Get handle of the kernel for urKernelGetNativeHandle.
  ze_kernel_handle_t getNativeZeHandle() const;

  // Get program handle of the kernel.
  ur_program_handle_t getProgramHandle() const;

  // Get name of the kernel.
  common_properties_t getCommonProperties() const;

  // Get properties of the kernel.
  const ze_kernel_properties_t &getProperties(ur_device_handle_t hDevice) const;

  // Implementation of urKernelSetArgValue.
  ur_result_t setArgValue(uint32_t argIndex, size_t argSize,
                          const ur_kernel_arg_value_properties_t *pProperties,
                          const void *pArgValue);

  // Implementation of urKernelSetArgPointer.
  ur_result_t
  setArgPointer(uint32_t argIndex,
                const ur_kernel_arg_pointer_properties_t *pProperties,
                const void *pArgValue);

  // Implementation of urKernelSetExecInfo.
  ur_result_t setExecInfo(ur_kernel_exec_info_t propName,
                          const void *pPropValue);

  std::vector<char> getSourceAttributes() const;

  // Perform cleanup.
  ur_result_t release();

  // Add a pending memory allocation for which device is not yet known.
  ur_result_t
  addPendingMemoryAllocation(pending_memory_allocation_t allocation);

  // Set all required values for the kernel before submission (including pending
  // memory allocations).
  ur_result_t
  prepareForSubmission(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                       const size_t *pGlobalWorkOffset, uint32_t workDim,
                       uint32_t groupSizeX, uint32_t groupSizeY,
                       uint32_t groupSizeZ,
                       std::function<void(void *, void *, size_t)> migrate);

private:
  // Keep the program of the kernel.
  const ur_program_handle_t hProgram;

  // Vector of ur_single_device_kernel_t indexed by deviceIndex().
  std::vector<std::optional<ur_single_device_kernel_t>> deviceKernels;

  // Cache of the common kernel properties.
  mutable ZeCache<common_properties_t> zeCommonProperties;

  // Index of the device in the deviceKernels vector.
  size_t deviceIndex(ur_device_handle_t hDevice) const;

  struct pending_memory_allocation_t {
    ur_mem_handle_t hMem;
    ur_mem_handle_t_::device_access_mode_t mode;
    uint32_t argIndex;
  };

  std::vector<pending_memory_allocation_t> pending_allocations;

  void completeInitialization();

  // pointer to any non-null kernel in deviceKernels
  ur_single_device_kernel_t *nonEmptyKernel;
};
