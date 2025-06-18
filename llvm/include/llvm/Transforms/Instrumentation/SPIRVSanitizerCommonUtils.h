//===- SPIRVSanitizerCommonUtils.h - Commnon utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common infrastructure for SPIRV Sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SPIRVSANITIZERCOMMONUTILS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SPIRVSANITIZERCOMMONUTILS_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
// Spir memory address space
constexpr unsigned kSpirOffloadPrivateAS = 0;
constexpr unsigned kSpirOffloadGlobalAS = 1;
constexpr unsigned kSpirOffloadConstantAS = 2;
constexpr unsigned kSpirOffloadLocalAS = 3;
constexpr unsigned kSpirOffloadGenericAS = 4;

TargetExtType *getTargetExtType(Type *Ty);
bool isJointMatrixAccess(Value *V);

// Sync with sanitizer_common/sanitizer_common.hpp
enum SanitizedKernelFlags : uint32_t {
  NO_CHECK = 0,
  CHECK_GLOBALS = 1U << 1,
  CHECK_LOCALS = 1U << 2,
  CHECK_PRIVATES = 1U << 3,
  CHECK_GENERICS = 1U << 4,
  MSAN_TRACK_ORIGINS = 1U << 5,
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SPIRVSANITIZERCOMMONUTILS_H
