//===- AMDGPUAddGlobalForAtomicXor.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Some AMDGPU atomic instructions require a prefetch in order for them to work
// properly when using hipMallocManaged. This pass scans a module for the
// problematic atomic instructions and creates a global PrefetchNeeded if the
// builtin is present. This allows the prefetch to happen at runtime only if the
// problematic builtin is chosen.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUAddGlobalForAtomicXor.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define NEW_GLOBAL_NAME "HipAtomicXorModuleNeedsPrefetch"

namespace {

bool moduleHasAtomicXor(Module &M) {
  for (auto &F : M) {
    for (auto &I : instructions(F)) {
      if (auto *AtomicInst = dyn_cast<AtomicRMWInst>(&I)) {
        if (AtomicInst->getOperation() == AtomicRMWInst::Xor) {
          return true;
        }
      }
    }
  }
  return false;
}

bool runImpl(Module &M) {
  if (moduleHasAtomicXor(M)) {
    LLVMContext &Ctx = M.getContext();
    new GlobalVariable(M, Type::getInt1Ty(Ctx), true,
                       GlobalValue::InternalLinkage,
                       Constant::getAllOnesValue(Type::getInt1Ty(Ctx)),
                       NEW_GLOBAL_NAME); // FIXME: this seems wrong and bad, is
                                         // there a better way to make a new
                                         // GlobalVariable?
    return true;
  }
  return false;
}
} // end anonymous namespace

PreservedAnalyses
AMDGPUAddGlobalForAtomicXorPass::run(Module &M, ModuleAnalysisManager &AM) {
  return runImpl(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
