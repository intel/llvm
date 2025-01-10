//===------- FAtomicsNativeCPU.cpp - Materializes FP Atomics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass that materializes floating points atomics by emitting
// corresponding atomicrmw instruction.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/FAtomicsNativeCPU.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"

using namespace llvm;

PreservedAnalyses FAtomicsNativeCPU::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  auto &Ctx = M.getContext();
  // TODO: add checks for windows mangling
  for (auto &F : M) {
    AtomicRMWInst::BinOp OpCode;
    if (F.getName().starts_with("_Z21__spirv_AtomicFAddEXT")) {
      OpCode = AtomicRMWInst::BinOp::FAdd;
    } else if (F.getName().starts_with("_Z21__spirv_AtomicFMinEXT")) {
      OpCode = AtomicRMWInst::BinOp::FMin;
    } else if (F.getName().starts_with("_Z21__spirv_AtomicFMaxEXT")) {
      OpCode = AtomicRMWInst::BinOp::FMax;
    } else {
      continue;
    }

    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", &F);
    IRBuilder<> Builder(BB);
    // Currently we drop arguments 1 and 2 (scope and memory ordering),
    // defaulting to Monotonic ordering and System scope.
    auto A =
        Builder.CreateAtomicRMW(OpCode, F.getArg(0), F.getArg(3), MaybeAlign(),
                                AtomicOrdering::Monotonic, SyncScope::System);
    Builder.CreateRet(A);
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
