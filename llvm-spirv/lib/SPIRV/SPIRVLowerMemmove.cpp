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

#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {

class SPIRVLowerMemmoveBase {
public:
  SPIRVLowerMemmoveBase() : Context(nullptr), Mod(nullptr) {}
  void LowerMemMoveInst(MemMoveInst &I) {
    IRBuilder<> Builder(I.getParent());
    Builder.SetInsertPoint(&I);
    auto *Dest = I.getRawDest();
    auto *Src = I.getRawSource();
    if (isa<PHINode>(Src))
      report_fatal_error("llvm.memmove of PHI instruction result not supported",
                         false);
    // The source could be bit-cast or addrspacecast from another type,
    // need the original type for the allocation of the temporary variable
    auto *SrcTy = Src->stripPointerCasts()->getType();
    auto *Length = cast<ConstantInt>(I.getLength());
    MaybeAlign Align = I.getSourceAlign();
    auto Volatile = I.isVolatile();
    Value *NumElements = nullptr;
    uint64_t ElementsCount = 1;
    if (SrcTy->isArrayTy()) {
      ElementsCount = SrcTy->getArrayNumElements();
      NumElements = Builder.getInt32(ElementsCount);
    }
    // Get number of bits to move and allocate memory appropriately:
    // if lenght is bigger than a pointer base type size, then create an
    // alloca of an array type with the same base type.
    const uint64_t LenBits = Length->getZExtValue();
    const uint64_t LayoutTypeBites =
        Mod->getDataLayout().getTypeSizeInBits(SrcTy->getPointerElementType()) *
        ElementsCount;
    auto *AllocaTy = SrcTy->getPointerElementType();
    if (LenBits > LayoutTypeBites) {
      const uint64_t ArraySize = LenBits / LayoutTypeBites;
      AllocaTy = ArrayType::get(SrcTy->getPointerElementType(), ArraySize);
    }
    auto *Alloca = Builder.CreateAlloca(AllocaTy, NumElements);
    if (Align.hasValue()) {
      Alloca->setAlignment(Align.getValue());
    }
    Builder.CreateLifetimeStart(Alloca);
    Builder.CreateMemCpy(Alloca, Align, Src, Align, Length, Volatile);
    auto *SecondCpy = Builder.CreateMemCpy(Dest, I.getDestAlign(), Alloca,
                                           Align, Length, Volatile);
    Builder.CreateLifetimeEnd(Alloca);

    SecondCpy->takeName(&I);
    I.replaceAllUsesWith(SecondCpy);
    I.dropAllReferences();
    I.eraseFromParent();
  }
  bool expandMemMoveIntrinsicUses(Function &F) {
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
  bool runLowerMemmove(Module &M) {
    Context = &M.getContext();
    Mod = &M;
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

private:
  LLVMContext *Context;
  Module *Mod;
};

class SPIRVLowerMemmovePass : public llvm::PassInfoMixin<SPIRVLowerMemmovePass>,
                              public SPIRVLowerMemmoveBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runLowerMemmove(M) ? llvm::PreservedAnalyses::none()
                              : llvm::PreservedAnalyses::all();
  }
};

class SPIRVLowerMemmoveLegacy : public ModulePass,
                                public SPIRVLowerMemmoveBase {
public:
  SPIRVLowerMemmoveLegacy() : ModulePass(ID) {
    initializeSPIRVLowerMemmoveLegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override { return runLowerMemmove(M); }

  static char ID;
};

char SPIRVLowerMemmoveLegacy::ID = 0;
} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerMemmoveLegacy, "spvmemmove",
                "Lower llvm.memmove into llvm.memcpy", false, false)

ModulePass *llvm::createSPIRVLowerMemmoveLegacy() {
  return new SPIRVLowerMemmoveLegacy();
}
