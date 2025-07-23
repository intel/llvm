//===--------- program.hpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

#include <atomic>
#include <unordered_map>

#include "common/ur_ref_count.hpp"
#include "context.hpp"

/// Implementation of UR Program on HIP Module object
struct ur_program_handle_t_ : ur::hip::handle_base {
  using native_type = hipModule_t;
  native_type Module;
  const char *Binary;
  size_t BinarySizeInBytes;
  ur::RefCount RefCount;
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  std::string ExecutableCache;

  // The ur_program_binary_type_t property is defined individually for every
  // device in a program. However, since the HIP adapter only has 1 device per
  // program, there is no need to keep track of its value for each
  // device.
  ur_program_binary_type_t BinaryType = UR_PROGRAM_BINARY_TYPE_NONE;

  // Metadata
  bool IsRelocatable = false;

  std::unordered_map<std::string, std::string> GlobalIDMD;
  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>>
      KernelReqdWorkGroupSizeMD;
  std::unordered_map<std::string, uint32_t> KernelReqdSubGroupSizeMD;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char ErrorLog[MAX_LOG_SIZE], InfoLog[MAX_LOG_SIZE];
  std::string BuildOptions;
  ur_program_build_status_t BuildStatus = UR_PROGRAM_BUILD_STATUS_NONE;

  ur_program_handle_t_(ur_context_handle_t Ctxt, ur_device_handle_t Device)
      : handle_base(), Module{nullptr}, Binary{}, BinarySizeInBytes{0},
        Context{Ctxt}, Device{Device}, KernelReqdWorkGroupSizeMD{},
        KernelReqdSubGroupSizeMD{} {
    urContextRetain(Context);

    // When the log is queried we use strnlen(InfoLog), so it needs to be
    // initialized like this when it's empty to correctly return 0.
    InfoLog[0] = '\0';
  }

  ~ur_program_handle_t_() { urContextRelease(Context); }

  ur_result_t setMetadata(const ur_program_metadata_t *Metadata, size_t Length);

  ur_result_t setBinary(const char *Binary, size_t BinarySizeInBytes);

  ur_result_t buildProgram(const char *BuildOptions);
  ur_result_t finalizeRelocatable();
  ur_context_handle_t getContext() const { return Context; };
  ur_device_handle_t getDevice() const { return Device; };

  native_type get() const noexcept { return Module; };

  ur_result_t getGlobalVariablePointer(const char *name,
                                       hipDeviceptr_t *DeviceGlobal,
                                       size_t *DeviceGlobalSize);
};
