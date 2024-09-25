//===-- SYCLLinkedModuleProcessor.cpp - finalize a fully linked module ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLLinkedModuleProcessor.h"

#include "llvm/Pass.h"

#define DEBUG_TYPE "sycl-linked-module-processor"
using namespace llvm;

namespace {
class SYCLLinkedModuleProcessor : public ModulePass {
public:
  static char ID;
  SYCLLinkedModuleProcessor(SpecConstantsPass::HandlingMode Mode)
      : ModulePass(ID), Mode(Mode) {
    initializeSYCLLinkedModuleProcessorPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    // TODO: determine if we need to run other passes
    ModuleAnalysisManager MAM;
    SpecConstantsPass SCP(Mode);
    auto PA = SCP.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SpecConstantsPass::HandlingMode Mode;
};
} // namespace
char SYCLLinkedModuleProcessor::ID = 0;
INITIALIZE_PASS(SYCLLinkedModuleProcessor, "SYCLLinkedModuleProcessor",
                "Finalize a fully linked SYCL module", false, false)
ModulePass *llvm::createSYCLLinkedModuleProcessorPass(
    SpecConstantsPass::HandlingMode Mode) {
  return new SYCLLinkedModuleProcessor(Mode);
}
