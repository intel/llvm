//===----- SpecConstants.h - SYCL Specialization Constants Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which converts symbolic id-based specialization
// constant intrinsics to integer id-based ones to later map to SPIRV spec
// constant operations. The spec constant IDs are symbolic before linkage to
// make separate compilation possible. After linkage all spec constants are
// available to the pass, and it can assign consistent integer IDs.
//
// The pass is used w/o a pass manager currently, but the interface is based on
// the standard Module pass interface to move it around easier in future.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

class SpecConstantsPass : public PassInfoMixin<SpecConstantsPass> {
public:
  // SetValAtRT parameter controls spec constant lowering mode:
  // - if true, it is lowered to SPIRV intrinsic which retrieves constant value
  // - if false, it is replaced with C++ default (used for AOT compilers)
  SpecConstantsPass(bool SetValAtRT = true) : SetValAtRT(SetValAtRT) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  // Searches given module for occurences of specialization constant-specific
  // metadata at call instructions and builds a
  // "spec constant name" -> "spec constant int ID" map from this information.
  static bool collectSpecConstantMetadata(Module &M,
                                          std::map<StringRef, unsigned> &IDMap);

private:
  bool SetValAtRT;
};
