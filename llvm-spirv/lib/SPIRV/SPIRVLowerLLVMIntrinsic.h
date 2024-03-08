//===- SPIRVLowerLLVMIntrinsic.h - llvm-intrinsic lowering  --------*- C++
//-*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2022 The Khronos Group Inc.
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
// Neither the names of The Khronos Group, nor the names of its
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

#ifndef SPIRV_SPIRVLOWERLLVMINTRINSIC_H
#define SPIRV_SPIRVLOWERLLVMINTRINSIC_H

#include "LLVMSPIRVOpts.h"

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace SPIRV {

class SPIRVLowerLLVMIntrinsicBase
    : public llvm::InstVisitor<SPIRVLowerLLVMIntrinsicBase> {
public:
  SPIRVLowerLLVMIntrinsicBase(const SPIRV::TranslatorOpts &Opts)
      : Context(nullptr), Mod(nullptr), Opts(Opts) {}
  virtual ~SPIRVLowerLLVMIntrinsicBase() {}
  virtual void visitIntrinsicInst(llvm::CallInst &I);

  bool runLowerLLVMIntrinsic(llvm::Module &M);

private:
  llvm::LLVMContext *Context;
  llvm::Module *Mod;
  const SPIRV::TranslatorOpts Opts;
  bool TheModuleIsModified = false;
};

class SPIRVLowerLLVMIntrinsicPass
    : public llvm::PassInfoMixin<SPIRVLowerLLVMIntrinsicPass>,
      public SPIRVLowerLLVMIntrinsicBase {
public:
  SPIRVLowerLLVMIntrinsicPass(const SPIRV::TranslatorOpts &Opts);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

class SPIRVLowerLLVMIntrinsicLegacy : public llvm::ModulePass,
                                      public SPIRVLowerLLVMIntrinsicBase {
public:
  SPIRVLowerLLVMIntrinsicLegacy(const SPIRV::TranslatorOpts &Opts);

  bool runOnModule(llvm::Module &M) override;

  static char ID;
};

} // namespace SPIRV

#endif // SPIRV_SPIRVLOWERLLVMINTRINSIC_H
