/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_options.hpp
 *
 */

#pragma once

#include "logger/ur_logger.hpp"
#include "ur/ur.hpp"

#include <cstdint>
#include <string>

namespace ur_sanitizer_layer {

struct SanitizerOptions {
  bool Debug = false;
  uint64_t MinRZSize = 16;
  uint64_t MaxQuarantineSizeMB = 8;
  bool DetectLocals = true;
  bool DetectPrivates = true;
  bool PrintStats = false;
  bool DetectKernelArguments = true;
  bool DetectLeaks = true;
  bool HaltOnError = true;
  bool Recover = false;

  void Init(const std::string &EnvName, logger::Logger &Logger);
};

} // namespace ur_sanitizer_layer
