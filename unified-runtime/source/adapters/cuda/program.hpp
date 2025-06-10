//===--------- program.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <ur_api.h>

#include <atomic>
#include <unordered_map>

#include "context.hpp"

struct ur_program_handle_t_ : ur::cuda::handle_base {
  using native_type = CUmodule;
  native_type Module;
  const char *Binary;
  size_t BinarySizeInBytes;
  std::atomic_uint32_t RefCount;
  ur_context_handle_t Context;
  ur_device_handle_t Device;

  /* The ur_program_binary_type_t property is defined individually for every
   * device in a program. However, since the CUDA adapter only has 1 device per
   * context / program, there is no need to keep track of its value for each
   * device. */
  ur_program_binary_type_t BinaryType = UR_PROGRAM_BINARY_TYPE_NONE;

  // Metadata
  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>>
      KernelReqdWorkGroupSizeMD;
  std::unordered_map<std::string, std::string> GlobalIDMD;
  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>>
      KernelMaxWorkGroupSizeMD;
  std::unordered_map<std::string, uint64_t> KernelMaxLinearWorkGroupSizeMD;
  std::unordered_map<std::string, uint32_t> KernelReqdSubGroupSizeMD;

  constexpr static size_t MaxLogSize = 8192u;

  char ErrorLog[MaxLogSize], InfoLog[MaxLogSize];
  std::string BuildOptions;
  ur_program_build_status_t BuildStatus = UR_PROGRAM_BUILD_STATUS_NONE;

  ur_program_handle_t_(ur_context_handle_t Context, ur_device_handle_t Device)
      : handle_base(), Module{nullptr}, Binary{}, BinarySizeInBytes{0},
        RefCount{1}, Context{Context}, Device{Device},
        KernelReqdWorkGroupSizeMD{}, KernelMaxWorkGroupSizeMD{},
        KernelMaxLinearWorkGroupSizeMD{}, KernelReqdSubGroupSizeMD{} {
    urContextRetain(Context);

    // When the log is queried we use strnlen(InfoLog), so it needs to be
    // initialized like this when it's empty to correctly return 0.
    InfoLog[0] = '\0';
  }

  ~ur_program_handle_t_() { urContextRelease(Context); }

  ur_result_t setMetadata(const ur_program_metadata_t *Metadata, size_t Length);

  ur_result_t setBinary(const char *Binary, size_t BinarySizeInBytes);

  ur_result_t buildProgram(const char *BuildOptions);
  ur_context_handle_t getContext() const { return Context; };
  ur_device_handle_t getDevice() const noexcept { return Device; };

  native_type get() const noexcept { return Module; };

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_result_t getGlobalVariablePointer(const char *name,
                                       CUdeviceptr *DeviceGlobal,
                                       size_t *DeviceGlobalSize);
};
