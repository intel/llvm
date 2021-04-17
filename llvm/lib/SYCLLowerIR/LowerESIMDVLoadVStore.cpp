//===- LowerESIMDVLoadVStore.cpp - lower vload/vstore to load/store -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// convert vload/vstore to load/store if they are not for genx_volatile.
//
// File scope simd variables marked with genx_volatile attribute want
// guarranteed allocation in register file, therefore we use vload/vstore
// instead of load/store, so they won't be optimized away by llvm.
//
// For ordinary simd variables, we do not need to protect load/store. But
// there is no good way to do this in clang. So we need this pass in the
// end of module passes to separate the cases that we need vload/vstore vs.
// the cases that we do not need vload/vstore
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loweresimdvloadvstore"

#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/LowerESIMD.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeESIMDLowerLoadStorePass(PassRegistry &);
}

namespace {

class ESIMDLowerLoadStore : public FunctionPass {
public:
  static char ID;
  ESIMDLowerLoadStore() : FunctionPass(ID) {
    initializeESIMDLowerLoadStorePass(*PassRegistry::getPassRegistry());
  }
  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  virtual bool runOnFunction(Function &F) override {
    FunctionAnalysisManager FAM;
    auto PA = Impl.run(F, FAM);
    return !PA.areAllPreserved();
  }

private:
  ESIMDLowerLoadStorePass Impl;
};

} // namespace

char ESIMDLowerLoadStore::ID = 0;
INITIALIZE_PASS(ESIMDLowerLoadStore, "ESIMDLowerLoadStore",
                "Lower ESIMD reference loads and stores", false, false)

// Lower non-volatilE vload/vstore intrinsic calls into normal load/store
// instructions.
PreservedAnalyses ESIMDLowerLoadStorePass::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  std::vector<Instruction *> ToErase;
  for (Instruction &Inst : instructions(F)) {
    if (!GenXIntrinsic::isVLoadStore(&Inst))
      continue;

    auto *Ptr = Inst.getOperand(0);
    if (GenXIntrinsic::isVStore(&Inst))
      Ptr = Inst.getOperand(1);
    auto AS0 = cast<PointerType>(Ptr->getType())->getAddressSpace();
    Ptr = Ptr->stripPointerCasts();
    auto GV = dyn_cast<GlobalVariable>(Ptr);
    if (!GV || !GV->hasAttribute("genx_volatile")) {
      // change to load/store
      IRBuilder<> Builder(&Inst);
      if (GenXIntrinsic::isVStore(&Inst))
        Builder.CreateStore(Inst.getOperand(0), Inst.getOperand(1));
      else {
        auto LI = Builder.CreateLoad(Inst.getType(), Inst.getOperand(0),
                                     Inst.getName());
        LI->setDebugLoc(Inst.getDebugLoc());
        Inst.replaceAllUsesWith(LI);
      }
      ToErase.push_back(&Inst);
    } else {
      // change to vload/vstore that has the same address space as
      // the global-var in order to clean up unnecessary addr-cast.
      auto AS1 = GV->getType()->getAddressSpace();
      if (AS0 != AS1) {
        IRBuilder<> Builder(&Inst);
        if (GenXIntrinsic::isVStore(&Inst)) {
          auto PtrTy = cast<PointerType>(Inst.getOperand(1)->getType());
          PtrTy = PointerType::get(PtrTy->getElementType(), AS1);
          auto PtrCast = Builder.CreateAddrSpaceCast(Inst.getOperand(1), PtrTy);
          Type *Tys[] = {Inst.getOperand(0)->getType(), PtrCast->getType()};
          Value *Args[] = {Inst.getOperand(0), PtrCast};
          Function *Fn = GenXIntrinsic::getGenXDeclaration(
              F.getParent(), GenXIntrinsic::genx_vstore, Tys);
          Builder.CreateCall(Fn, Args, Inst.getName());
        } else {
          auto PtrTy = cast<PointerType>(Inst.getOperand(0)->getType());
          PtrTy = PointerType::get(PtrTy->getElementType(), AS1);
          auto PtrCast = Builder.CreateAddrSpaceCast(Inst.getOperand(0), PtrTy);
          Type *Tys[] = {Inst.getType(), PtrCast->getType()};
          Function *Fn = GenXIntrinsic::getGenXDeclaration(
              F.getParent(), GenXIntrinsic::genx_vload, Tys);
          Value *VLoad = Builder.CreateCall(Fn, PtrCast, Inst.getName());
          Inst.replaceAllUsesWith(VLoad);
        }
        ToErase.push_back(&Inst);
      }
    }
  }

  for (auto Inst : ToErase) {
    Inst->eraseFromParent();
  }

  return !ToErase.empty() ? PreservedAnalyses::none()
                          : PreservedAnalyses::all();
}

namespace llvm {
FunctionPass *createESIMDLowerLoadStorePass() {
  return new ESIMDLowerLoadStore;
}
} // namespace llvm
