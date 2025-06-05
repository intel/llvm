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

#include "llvm/ADT/SmallString.h"
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

SmallString<128> computeKernelMetadataUniqueId(StringRef Prefix,
                              SmallVectorImpl<uint8_t> &KernelNamesBytes);
} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SPIRVSANITIZERCOMMONUTILS_H
