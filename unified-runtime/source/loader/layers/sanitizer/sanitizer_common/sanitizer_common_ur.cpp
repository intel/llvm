
/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_common_ur.cpp
 *
 * This file contains the common functions and data structures related to UR.
 *
 */

#include "ur/ur.hpp"
#include "ur_sanitizer_layer.hpp"
namespace ur_sanitizer_layer {

void PrintUrBuildLog(ur_program_handle_t hProgram,
                     ur_device_handle_t *phDevices, size_t numDevices) {
  getContext()->logger.debug("Printing build log for program {}", hProgram);
  for (size_t i = 0; i < numDevices; i++) {
    std::vector<char> LogBuf;
    size_t LogSize = 0;
    auto hDevice = phDevices[i];

    auto UrRes = getContext()->urDdiTable.Program.pfnGetBuildInfo(
        hProgram, hDevice, UR_PROGRAM_BUILD_INFO_LOG, 0, nullptr, &LogSize);
    if (UrRes != UR_RESULT_SUCCESS) {
      getContext()->logger.debug("For device {}: failed to get build log size.",
                                 hDevice);
      continue;
    }

    LogBuf.resize(LogSize);
    UrRes = getContext()->urDdiTable.Program.pfnGetBuildInfo(
        hProgram, hDevice, UR_PROGRAM_BUILD_INFO_LOG, LogSize, LogBuf.data(),
        nullptr);
    if (UrRes != UR_RESULT_SUCCESS) {
      getContext()->logger.debug("For device {}: failed to get build log.",
                                 hDevice);
      continue;
    }

    getContext()->logger.debug("For device {}:\n{}", hDevice, LogBuf.data());
  }
}

} // namespace ur_sanitizer_layer
