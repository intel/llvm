//===- OCLTypeToSPIRV.h - Adapt types from OCL for SPIRV --------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements adaptation of OCL types for SPIRV. It does not modify
// the module. Instead, it returns adapted function type based on kernel
// argument metadata. Later LLVM/SPIRV translator will translate the adapted
// type instead of the original type.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_OCLTYPETOSPIRV_H
#define SPIRV_OCLTYPETOSPIRV_H

#include "LLVMSPIRVLib.h"
#include "SPIRVBuiltinHelper.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include <map>
#include <set>

namespace SPIRV {

class OCLTypeToSPIRVBase : protected BuiltinCallHelper {
public:
  OCLTypeToSPIRVBase();

  bool runOCLTypeToSPIRV(llvm::Module &M);

  /// Returns the adapted type of the corresponding argument for a function. If
  /// the type is a pointer type, it will return a TypedPointerType instead.
  llvm::Type *getAdaptedArgumentType(llvm::Function *F, unsigned ArgNo);

private:
  llvm::Module *M;
  llvm::LLVMContext *Ctx;
  // Map of argument/Function -> adapted type (probably TypedPointerType)
  std::unordered_map<llvm::Value *, llvm::Type *> AdaptedTy;
  std::set<llvm::Function *> WorkSet; // Functions to be adapted

  void adaptFunctionArguments(llvm::Function *F);
  void adaptArgumentsByMetadata(llvm::Function *F);
  void adaptArgumentsBySamplerUse(llvm::Module &M);
  void adaptFunction(llvm::Function *F);
  void addAdaptedType(llvm::Value *V, llvm::Type *Ty);
  void addWork(llvm::Function *F);
};

class OCLTypeToSPIRVLegacy : public OCLTypeToSPIRVBase,
                             public llvm::ModulePass {
public:
  OCLTypeToSPIRVLegacy();
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  bool runOnModule(llvm::Module &M) override;
  static char ID;
};

class OCLTypeToSPIRVPass : public OCLTypeToSPIRVBase,
                           public llvm::AnalysisInfoMixin<OCLTypeToSPIRVPass> {
public:
  using Result = OCLTypeToSPIRVBase;
  static llvm::AnalysisKey Key;
  OCLTypeToSPIRVBase &run(llvm::Module &F, llvm::ModuleAnalysisManager &MAM);
};

} // namespace SPIRV

#endif // SPIRV_OCLTYPETOSPIRV_H
