//===- SanitizerCommonUtils.h - Sanitizer commnon utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common infrastructure for Sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERCOMMONUTILS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERCOMMONUTILS_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
namespace SanitizerCommonUtils {

// Spir memory address space
constexpr unsigned kSpirOffloadPrivateAS = 0;
constexpr unsigned kSpirOffloadGlobalAS = 1;
constexpr unsigned kSpirOffloadConstantAS = 2;
constexpr unsigned kSpirOffloadLocalAS = 3;
constexpr unsigned kSpirOffloadGenericAS = 4;

TargetExtType *getTargetExtType(Type *Ty);
bool isJointMatrixAccess(Value *V);
} // namespace SanitizerCommonUtils
} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SANITIZERCOMMONUTILS_H
