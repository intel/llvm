/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_options.cpp
 *
 */

#include "sanitizer_options.hpp"
#include "sanitizer_common.hpp"
#include "sanitizer_options_impl.hpp"

#include <cstring>
#include <stdexcept>

namespace ur_sanitizer_layer {

void SanitizerOptions::Init(const std::string &EnvName,
                            logger::Logger &Logger) {
  std::optional<EnvVarMap> OptionsEnvMap;
  try {
    OptionsEnvMap = getenv_to_map(EnvName.c_str());
  } catch (const std::invalid_argument &e) {
    std::stringstream SS;
    SS << "<SANITIZER>[ERROR]: ";
    SS << e.what();
    UR_LOG_L(Logger, QUIET, SS.str().c_str());
    die("Sanitizer failed to parse options.\n");
  }

  if (!OptionsEnvMap.has_value()) {
    return;
  }

  auto Parser = options::OptionParser(OptionsEnvMap.value(), Logger);

  Parser.ParseBool("debug", Debug);
  Parser.ParseBool("detect_kernel_arguments", DetectKernelArguments);
  Parser.ParseBool("detect_locals", DetectLocals);
  Parser.ParseBool("detect_privates", DetectPrivates);
  Parser.ParseBool("print_stats", PrintStats);
  Parser.ParseBool("detect_leaks", DetectLeaks);
  Parser.ParseBool("halt_on_error", HaltOnError);
  Parser.ParseBool("recover", Recover);

  Parser.ParseUint64("quarantine_size_mb", MaxQuarantineSizeMB, 0, UINT32_MAX);
  Parser.ParseUint64("redzone", MinRZSize, 16);
  MinRZSize =
      IsPowerOfTwo(MinRZSize) ? MinRZSize : RoundUpToPowerOfTwo(MinRZSize);
  if (MinRZSize > 16) {
    UR_LOG_L(Logger, WARN,
             "Increasing the redzone size may cause excessive memory overhead");
  }
}

} // namespace ur_sanitizer_layer
