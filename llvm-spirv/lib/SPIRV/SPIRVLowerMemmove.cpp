//===- SPIRVLowerMemmove.cpp - Lower llvm.memmove to llvm.memcpys ---------===//
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
// This file implements lowering llvm.memmove into several llvm.memcpys.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvmemmove"

#include "SPIRVLowerMemmove.h"
#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {

void SPIRVLowerMemmoveBase::LowerMemMoveInst(MemMoveInst &I) {
  // There is no direct equivalent of @llvm.memmove in SPIR-V and the closest
  // instructions are 'OpCopyMemory' and 'OpCopyMemorySized'.
  //
  // 'OpCopyMemory' does not accept amount of bytes to copy and infers that
  // from type which is being copied; also it only allows to copy value of a
  // particular type to pointer pointing to the same type.
  //
  // 'OpCopyMemorySized' is closer to @llvm.memmove, because it actually
  // copies bytes, but unlike memove it is not explicitly specified whether it
  // supports overlapping source and destination. Therefore, we replace
  // memmove with two 'OpCopyMemorySized' instructions: the first one copies
  // bytes from source to a temporary location, the second one copies bytes
  // from that temporary location to the destination.
  IRBuilder<> Builder(I.getParent());
  Builder.SetInsertPoint(&I);

  auto *Length = cast<ConstantInt>(I.getLength());
  auto *AllocaTy =
      ArrayType::get(IntegerType::getInt8Ty(*Context), Length->getZExtValue());
  MaybeAlign SrcAlign = I.getSourceAlign();

  auto *Alloca = Builder.CreateAlloca(AllocaTy);
  if (SrcAlign.hasValue())
    Alloca->setAlignment(SrcAlign.getValue());

  // FIXME: Do we need to pass the size of alloca here? From LangRef:
  // > The first argument is a constant integer representing the size of the
  // > object, or -1 if it is variable sized.
  //
  // https://llvm.org/docs/LangRef.html#llvm-lifetime-start-intrinsic
  Builder.CreateLifetimeStart(Alloca);
  Builder.CreateMemCpy(Alloca, SrcAlign, I.getRawSource(), SrcAlign, Length,
                       I.isVolatile());

  auto *SecondCpy =
      Builder.CreateMemCpy(I.getRawDest(), I.getDestAlign(), Alloca, SrcAlign,
                           Length, I.isVolatile());
  Builder.CreateLifetimeEnd(Alloca);

  SecondCpy->takeName(&I);
  I.replaceAllUsesWith(SecondCpy);
  I.dropAllReferences();
  I.eraseFromParent();
}

bool SPIRVLowerMemmoveBase::expandMemMoveIntrinsicUses(Function &F) {
  bool Changed = false;

  for (User *U : make_early_inc_range(F.users())) {
    MemMoveInst *Inst = cast<MemMoveInst>(U);
    if (!isa<ConstantInt>(Inst->getLength())) {
      expandMemMoveAsLoop(Inst);
      Inst->eraseFromParent();
    } else {
      LowerMemMoveInst(*Inst);
    }
    Changed = true;
  }
  return Changed;
}

bool SPIRVLowerMemmoveBase::runLowerMemmove(Module &M) {
  Context = &M.getContext();
  bool Changed = false;

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    if (F.getIntrinsicID() == Intrinsic::memmove)
      Changed |= expandMemMoveIntrinsicUses(F);
  }

  verifyRegularizationPass(M, "SPIRVLowerMemmove");
  return Changed;
}

llvm::PreservedAnalyses
SPIRVLowerMemmovePass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  return runLowerMemmove(M) ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all();
}

SPIRVLowerMemmoveLegacy::SPIRVLowerMemmoveLegacy() : ModulePass(ID) {
  initializeSPIRVLowerMemmoveLegacyPass(*PassRegistry::getPassRegistry());
}

bool SPIRVLowerMemmoveLegacy::runOnModule(Module &M) {
  return runLowerMemmove(M);
}

char SPIRVLowerMemmoveLegacy::ID = 0;

} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerMemmoveLegacy, "spvmemmove",
                "Lower llvm.memmove into llvm.memcpy", false, false)

ModulePass *llvm::createSPIRVLowerMemmoveLegacy() {
  return new SPIRVLowerMemmoveLegacy();
}
