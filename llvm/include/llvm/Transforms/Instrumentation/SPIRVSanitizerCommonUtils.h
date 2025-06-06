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
#include "llvm/IR/Constants.h"
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

// If the type is or has target extension type just return the type, otherwise
// return nullptr.
TargetExtType *getTargetExtType(Type *Ty);

// Check if it's a joint matrix access operation.
bool isJointMatrixAccess(Value *V);

// If the User is an instruction of constant expr, try to get the functions that
// it has been used.
void getFunctionsOfUser(User *User, SmallVectorImpl<Function *> &Functions);

// Compute MD5 hash for kernel metadata global as unique id.
SmallString<128>
computeKernelMetadataUniqueId(StringRef Prefix,
                              SmallVectorImpl<uint8_t> &KernelNamesBytes);

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SPIRVSANITIZERCOMMONUTILS_H
