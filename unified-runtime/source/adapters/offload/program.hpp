//===----------- program.hpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <ur_api.h>

#include "common.hpp"

struct ur_program_handle_t_ : RefCounted {
  ol_program_handle_t OffloadProgram;
  ur_context_handle_t URContext;
  const uint8_t *Binary;
  size_t BinarySizeInBytes;
  // A mapping from mangled global names -> names in the binary
  std::unordered_map<std::string, std::string> GlobalIDMD;
  // The UR offload backend doesn't draw distinctions between these types (we
  // always have a fully built binary), but we need to track what state we are
  // pretending to be in
  ur_program_binary_type_t BinaryType;
  std::string Error;

  static ur_program_handle_t_ *newErrorProgram(ur_context_handle_t Context,
                                               const uint8_t *Binary,
                                               size_t BinarySizeInBytes,
                                               std::string &&Error) {
    return new ur_program_handle_t_{{},
                                    nullptr,
                                    Context,
                                    Binary,
                                    BinarySizeInBytes,
                                    {},
                                    UR_PROGRAM_BINARY_TYPE_NONE,
                                    Error};
  }
};
