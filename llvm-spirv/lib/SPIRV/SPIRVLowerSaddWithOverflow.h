//===- SPIRVLowerSaddWithOverflowBase.h - sadd lowering  --------*- C++ -*-===//
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

#ifndef SPIRV_SPIRVLOWERSADDWITHOVERFLOW_H
#define SPIRV_SPIRVLOWERSADDWITHOVERFLOW_H

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace SPIRV {

class SPIRVLowerSaddWithOverflowBase
    : public llvm::InstVisitor<SPIRVLowerSaddWithOverflowBase> {
public:
  SPIRVLowerSaddWithOverflowBase() : Context(nullptr), Mod(nullptr) {}
  virtual ~SPIRVLowerSaddWithOverflowBase() {}
  virtual void visitIntrinsicInst(llvm::CallInst &I);

  bool runLowerSaddWithOverflow(llvm::Module &M);

private:
  llvm::LLVMContext *Context;
  llvm::Module *Mod;
  bool TheModuleIsModified = false;
};

class SPIRVLowerSaddWithOverflowPass
    : public llvm::PassInfoMixin<SPIRVLowerSaddWithOverflowPass>,
      public SPIRVLowerSaddWithOverflowBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

class SPIRVLowerSaddWithOverflowLegacy : public llvm::ModulePass,
                                         public SPIRVLowerSaddWithOverflowBase {
public:
  SPIRVLowerSaddWithOverflowLegacy();

  bool runOnModule(llvm::Module &M) override;

  static char ID;
};

} // namespace SPIRV

#endif // SPIRV_SPIRVLOWERSADDWITHOVERFLOW_H
