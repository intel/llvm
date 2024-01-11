//=- PreprocessMetadata.h - Metadata preprocessing pass -*- C++ -*-=//
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
//
// This file implements preprocessing of LLVM IR metadata in order to perform
// further translation to SPIR-V.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_PREPROCESSMETADATA_H
#define SPIRV_PREPROCESSMETADATA_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "SPIRVMDBuilder.h"

namespace SPIRV {

class SPIRVMDWalker;

class PreprocessMetadataBase {
public:
  PreprocessMetadataBase() : M(nullptr), Ctx(nullptr) {}

  bool runPreprocessMetadata(Module &M);
  void visit(Module *M);
  void preprocessCXXStructorList(SPIRVMDBuilder::NamedMDWrapper &EM,
                                 GlobalVariable *V, ExecutionMode EMode);
  void preprocessOCLMetadata(Module *M, SPIRVMDBuilder *B, SPIRVMDWalker *W);
  void preprocessVectorComputeMetadata(Module *M, SPIRVMDBuilder *B,
                                       SPIRVMDWalker *W);

private:
  Module *M;
  LLVMContext *Ctx;
};

class PreprocessMetadataLegacy : public ModulePass,
                                 public PreprocessMetadataBase {
public:
  PreprocessMetadataLegacy() : ModulePass(ID) {
    initializePreprocessMetadataLegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;

  static char ID;
};

class PreprocessMetadataPass
    : public llvm::PassInfoMixin<PreprocessMetadataPass>,
      public PreprocessMetadataBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace SPIRV

#endif // SPIRV_PREPROCESSMETADATA_H
