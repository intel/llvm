//==----- SYCLDeviceLibBF16.cpp - get SYCL BF16 devicelib required Info ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file provides some utils to analyze whether user's device image does
// depend on sycl bfloat16 device library functions.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLDeviceLibBF16.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

static constexpr char DEVICELIB_FUNC_PREFIX[] = "__devicelib_";

using namespace llvm;

static llvm::SmallVector<StringRef, 14> BF16DeviceLibFuncs = {
    "__devicelib_ConvertFToBF16INTEL",
    "__devicelib_ConvertBF16ToFINTEL",
    "__devicelib_ConvertFToBF16INTELVec1",
    "__devicelib_ConvertBF16ToFINTELVec1",
    "__devicelib_ConvertFToBF16INTELVec2",
    "__devicelib_ConvertBF16ToFINTELVec2",
    "__devicelib_ConvertFToBF16INTELVec3",
    "__devicelib_ConvertBF16ToFINTELVec3",
    "__devicelib_ConvertFToBF16INTELVec4",
    "__devicelib_ConvertBF16ToFINTELVec4",
    "__devicelib_ConvertFToBF16INTELVec8",
    "__devicelib_ConvertBF16ToFINTELVec8",
    "__devicelib_ConvertFToBF16INTELVec16",
    "__devicelib_ConvertBF16ToFINTELVec16",
};

bool llvm::isSYCLDeviceLibBF16Used(const Module &M) {
  if (!Triple(M.getTargetTriple()).isSPIROrSPIRV())
    return false;

  for (auto Fn : BF16DeviceLibFuncs) {
    Function *BF16Func = M.getFunction(Fn);
    if (BF16Func && BF16Func->isDeclaration())
      return true;
  }

  return false;
}

bool llvm::isBF16DeviceLibFuncDecl(const Function &F) {
  if (!F.isDeclaration() || !F.getName().starts_with(DEVICELIB_FUNC_PREFIX))
    return false;
  for (auto BFunc : BF16DeviceLibFuncs) {
    if (!F.getName().compare(BFunc))
      return true;
  }

  return false;
}
