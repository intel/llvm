//===- SPIRVRegularizeLLVM.cpp - Regularize LLVM for SPIR-V ------- C++ -*-===//
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
// This file implements regularization of LLVM module for SPIR-V.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvregular"

#include "OCLUtil.h"
#include "SPIRVInternal.h"

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <vector>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

static bool SPIRVDbgSaveRegularizedModule = false;
static std::string RegularizedModuleTmpFile = "regularized.bc";

class SPIRVRegularizeLLVM : public ModulePass {
public:
  SPIRVRegularizeLLVM() : ModulePass(ID), M(nullptr), Ctx(nullptr) {
    initializeSPIRVRegularizeLLVMPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  // Lower functions
  bool regularize();

  /// Erase cast inst of function and replace with the function.
  /// Assuming F is a SPIR-V builtin function with op code \param OC.
  void lowerFuncPtr(Function *F, Op OC);
  void lowerFuncPtr(Module *M);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
};

char SPIRVRegularizeLLVM::ID = 0;

bool SPIRVRegularizeLLVM::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter SPIRVRegularizeLLVM:\n");
  regularize();

  LLVM_DEBUG(dbgs() << "After SPIRVRegularizeLLVM:\n" << *M);
  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

/// Remove entities not representable by SPIR-V
bool SPIRVRegularizeLLVM::regularize() {
  eraseUselessFunctions(M);
  lowerFuncPtr(M);

  for (auto I = M->begin(), E = M->end(); I != E;) {
    Function *F = &(*I++);
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      continue;
    }

    std::vector<Instruction *> ToErase;
    for (BasicBlock &BB : *F) {
      for (Instruction &II : BB) {
        if (auto Call = dyn_cast<CallInst>(&II)) {
          Call->setTailCall(false);
          Function *CF = Call->getCalledFunction();
          if (CF && CF->isIntrinsic())
            removeFnAttr(Call, Attribute::NoUnwind);
        }

        // Remove optimization info not supported by SPIRV
        if (auto BO = dyn_cast<BinaryOperator>(&II)) {
          if (isa<PossiblyExactOperator>(BO) && BO->isExact())
            BO->setIsExact(false);
        }
        // Remove metadata not supported by SPIRV
        static const char *MDs[] = {
            "fpmath",
            "tbaa",
            "range",
        };
        for (auto &MDName : MDs) {
          if (II.getMetadata(MDName)) {
            II.setMetadata(MDName, nullptr);
          }
        }
        if (auto Cmpxchg = dyn_cast<AtomicCmpXchgInst>(&II)) {
          Value *Ptr = Cmpxchg->getPointerOperand();
          // To get memory scope argument we might use Cmpxchg->getSyncScopeID()
          // but LLVM's cmpxchg instruction is not aware of OpenCL(or SPIR-V)
          // memory scope enumeration. And assuming the produced SPIR-V module
          // will be consumed in an OpenCL environment, we can use the same
          // memory scope as OpenCL atomic functions that do not have
          // memory_scope argument, i.e. memory_scope_device. See the OpenCL C
          // specification p6.13.11. Atomic Functions
          Value *MemoryScope = getInt32(M, spv::ScopeDevice);
          auto SuccessOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getSuccessOrdering()));
          auto FailureOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getFailureOrdering()));
          Value *EqualSem = getInt32(M, OCLMemOrderMap::map(SuccessOrder));
          Value *UnequalSem = getInt32(M, OCLMemOrderMap::map(FailureOrder));
          Value *Val = Cmpxchg->getNewValOperand();
          Value *Comparator = Cmpxchg->getCompareOperand();

          llvm::Value *Args[] = {Ptr,        MemoryScope, EqualSem,
                                 UnequalSem, Val,         Comparator};
          auto *Res = addCallInstSPIRV(M, "__spirv_AtomicCompareExchange",
                                       Cmpxchg->getCompareOperand()->getType(),
                                       Args, nullptr, &II, "cmpxchg.res");
          // cmpxchg LLVM instruction returns a pair: the original value and
          // a flag indicating success (true) or failure (false).
          // OpAtomicCompareExchange SPIR-V instruction returns only the
          // original value. So we replace all uses of the original value
          // extracted from the pair with the result of OpAtomicCompareExchange
          // instruction. And we replace all uses of the flag with result of an
          // OpIEqual instruction. The OpIEqual instruction returns true if the
          // original value equals to the comparator which matches with
          // semantics of cmpxchg.
          for (User *U : Cmpxchg->users()) {
            if (auto *Extract = dyn_cast<ExtractValueInst>(U)) {
              if (Extract->getIndices()[0] == 0) {
                Extract->replaceAllUsesWith(Res);
              } else if (Extract->getIndices()[0] == 1) {
                auto *Cmp = new ICmpInst(Extract, CmpInst::ICMP_EQ, Res,
                                         Comparator, "cmpxchg.success");
                Extract->replaceAllUsesWith(Cmp);
              } else {
                llvm_unreachable("Unxpected cmpxchg pattern");
              }
              assert(Extract->user_empty());
              Extract->dropAllReferences();
              ToErase.push_back(Extract);
            }
          }
          if (Cmpxchg->user_empty())
            ToErase.push_back(Cmpxchg);
        }
      }
    }
    for (Instruction *V : ToErase) {
      assert(V->user_empty());
      V->eraseFromParent();
    }
  }

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    SPIRVDBG(errs() << "Fails to verify module: " << ErrorOS.str();)
    return false;
  }

  if (SPIRVDbgSaveRegularizedModule)
    saveLLVMModule(M, RegularizedModuleTmpFile);
  return true;
}

// Assume F is a SPIR-V builtin function with a function pointer argument which
// is a bitcast instruction casting a function to a void(void) function pointer.
void SPIRVRegularizeLLVM::lowerFuncPtr(Function *F, Op OC) {
  LLVM_DEBUG(dbgs() << "[lowerFuncPtr] " << *F << '\n');
  auto Name = decorateSPIRVFunction(getName(OC));
  std::set<Value *> InvokeFuncPtrs;
  auto Attrs = F->getAttributes();
  mutateFunction(
      F,
      [=, &InvokeFuncPtrs](CallInst *CI, std::vector<Value *> &Args) {
        for (auto &I : Args) {
          if (isFunctionPointerType(I->getType())) {
            InvokeFuncPtrs.insert(I);
            I = removeCast(I);
          }
        }
        return Name;
      },
      nullptr, &Attrs, false);
  for (auto &I : InvokeFuncPtrs)
    eraseIfNoUse(I);
}

void SPIRVRegularizeLLVM::lowerFuncPtr(Module *M) {
  std::vector<std::pair<Function *, Op>> Work;
  for (auto &F : *M) {
    auto AI = F.arg_begin();
    if (hasFunctionPointerArg(&F, AI)) {
      auto OC = getSPIRVFuncOC(F.getName());
      if (OC != OpNop) // builtin with a function pointer argument
        Work.push_back(std::make_pair(&F, OC));
    }
  }
  for (auto &I : Work)
    lowerFuncPtr(I.first, I.second);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVRegularizeLLVM, "spvregular", "Regularize LLVM for SPIR-V",
                false, false)

ModulePass *llvm::createSPIRVRegularizeLLVM() {
  return new SPIRVRegularizeLLVM();
}
