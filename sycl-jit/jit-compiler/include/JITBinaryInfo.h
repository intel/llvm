//===- JTIBinaryInfo.h - Non-owning descriptor for a binary image ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace jit_compiler {

using BinaryAddress = const uint8_t *;

/// Different binary formats supported as input to/output from the JIT compiler.
enum class BinaryFormat : uint32_t { INVALID, LLVM, SPIRV, PTX, AMDGCN };

/// Non-owning descriptor for a device intermediate representation module (e.g.,
/// SPIR-V, LLVM IR) from DPC++.
struct JITBinaryInfo {
  BinaryFormat Format = BinaryFormat::INVALID;

  uint64_t AddressBits = 0;

  BinaryAddress BinaryStart = nullptr;

  uint64_t BinarySize = 0;
};

} // namespace jit_compiler
