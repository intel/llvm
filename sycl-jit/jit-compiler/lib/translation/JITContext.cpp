//===- JITContext.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITContext.h"

using namespace jit_compiler;

#ifndef _WIN32
JITContext *JITContext::Instance = nullptr;

void JITContext::destroyInstance() {
  delete Instance;
  Instance = nullptr;
}

// Match the runtime's late unload priority, after user static destructors.
__attribute__((destructor(110))) static void jitUnload() {
  JITContext::destroyInstance();
}
#endif

JITBinary::JITBinary(std::string &&Binary, BinaryFormat Fmt)
    : Blob{std::move(Binary)}, Format{Fmt} {}

jit_compiler::BinaryAddress JITBinary::address() const {
  // The `reinterpret_cast` is deemed safe here because `JITBinary` instances
  // cannot be copied or moved, hence the `Blob` member will remain unmodified
  // during the object's lifetime.
  return reinterpret_cast<jit_compiler::BinaryAddress>(Blob.c_str());
}

size_t JITBinary::size() const { return Blob.size(); }

BinaryFormat JITBinary::format() const { return Format; }
