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

struct ur_single_device_kernel_t {
  ur_single_device_kernel_t(ze_device_handle_t hDevice,
                            ze_kernel_handle_t hKernel, bool ownZeHandle);
  ur_result_t release();

  ze_device_handle_t hDevice;
  v2::raii::ze_kernel_handle_t hKernel;
  mutable ZeCache<ZeStruct<ze_kernel_properties_t>> zeKernelProperties;
};

struct ur_kernel_handle_t_ : _ur_object {
private:
public:
  ur_kernel_handle_t_(ur_program_handle_t hProgram, const char *kernelName);

  // From native handle
  ur_kernel_handle_t_(ur_native_handle_t hNativeKernel,
                      ur_program_handle_t hProgram,
                      const ur_kernel_native_properties_t *pProperties);

  // Get L0 kernel handle for a given device
  ze_kernel_handle_t getZeHandle(ur_device_handle_t hDevice);

  // Get program handle of the kernel.
  ur_program_handle_t getProgramHandle() const;

  // Get name of the kernel.
  const std::string &getName() const;

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

  // Perform cleanup.
  ur_result_t release();

private:
  // Keep the program of the kernel.
  ur_program_handle_t hProgram;

  // Vector of ur_single_device_kernel_t indexed by device->Id
  std::vector<std::optional<ur_single_device_kernel_t>> deviceKernels;

  // Cache of the kernel name.
  mutable ZeCache<std::string> zeKernelName;

  void completeInitialization();
};
