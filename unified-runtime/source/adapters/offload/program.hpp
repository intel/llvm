//===----------- program.hpp - LLVM Offload Adapter  ----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <unified-runtime/ur_api.h>

#include "common.hpp"

struct ur_program_handle_t_ : RefCounted {
  explicit ur_program_handle_t_(ur_context_handle_t Context)
      : URContext(Context) {
    urContextRetain(URContext);
  }

  ~ur_program_handle_t_() { urContextRelease(URContext); }

  ol_program_handle_t OffloadProgram = nullptr;
  ur_context_handle_t URContext;
  const uint8_t *Binary = nullptr;
  size_t BinarySizeInBytes = 0;
  // A mapping from mangled global names -> names in the binary
  std::unordered_map<std::string, std::string> GlobalIDMD;
  // The UR offload backend doesn't draw distinctions between these types (we
  // always have a fully built binary), but we need to track what state we are
  // pretending to be in
  ur_program_binary_type_t BinaryType = UR_PROGRAM_BINARY_TYPE_NONE;
  std::string Error;

  static ur_program_handle_t_ *newErrorProgram(ur_context_handle_t Context,
                                               const uint8_t *Binary,
                                               size_t BinarySizeInBytes,
                                               std::string &&Error) {
    auto *Program = new ur_program_handle_t_(Context);
    Program->Binary = Binary;
    Program->BinarySizeInBytes = BinarySizeInBytes;
    Program->Error = std::move(Error);
    return Program;
  }
};
