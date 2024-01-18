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

#include "context.hpp"

/// Implementation of UR Program on HIP Module object
struct ur_program_handle_t_ {
  using native_type = hipModule_t;
  native_type Module;
  const char *Binary;
  size_t BinarySizeInBytes;
  std::atomic_uint32_t RefCount;
  ur_context_handle_t Context;
  ur_device_handle_t Device;
  std::string ExecutableCache;

  // Metadata
  bool IsRelocatable = false;

  std::unordered_map<std::string, std::string> GlobalIDMD;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char ErrorLog[MAX_LOG_SIZE], InfoLog[MAX_LOG_SIZE];
  std::string BuildOptions;
  ur_program_build_status_t BuildStatus = UR_PROGRAM_BUILD_STATUS_NONE;

  ur_program_handle_t_(ur_context_handle_t Ctxt, ur_device_handle_t Device)
      : Module{nullptr}, Binary{},
        BinarySizeInBytes{0}, RefCount{1}, Context{Ctxt}, Device{Device} {
    urContextRetain(Context);
    urDeviceRetain(Device);
  }

  ~ur_program_handle_t_() {
    urContextRelease(Context);
    urDeviceRelease(Device);
  }

  ur_result_t setMetadata(const ur_program_metadata_t *Metadata, size_t Length);

  ur_result_t setBinary(const char *Binary, size_t BinarySizeInBytes);

  ur_result_t buildProgram(const char *BuildOptions);
  ur_result_t finalizeRelocatable();
  ur_context_handle_t getContext() const { return Context; };
  ur_device_handle_t getDevice() const { return Device; };

  native_type get() const noexcept { return Module; };

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
