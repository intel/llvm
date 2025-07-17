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
};
