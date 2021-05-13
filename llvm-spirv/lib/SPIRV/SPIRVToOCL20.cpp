//===- SPIRVToOCL20.cpp - Transform SPIR-V builtins to OCL20 builtins------===//
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
// This file implements transform SPIR-V builtins to OCL 2.0 builtins.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvtocl20"

#include "OCLUtil.h"
#include "SPIRVToOCL.h"
#include "llvm/IR/Verifier.h"

namespace SPIRV {

class SPIRVToOCL20Base : public SPIRVToOCLBase {
public:
  bool runSPIRVToOCL(Module &M) override;

  /// Transform __spirv_MemoryBarrier to atomic_work_item_fence.
  ///   __spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  void visitCallSPIRVMemoryBarrier(CallInst *CI) override;

  /// Transform __spirv_ControlBarrier to work_group_barrier/sub_group_barrier.
  /// If execution scope is ScopeWorkgroup:
  ///    __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///         work_group_barrier(flag(sema), map(memScope))
  /// Otherwise:
  ///    __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///         sub_group_barrier(flag(sema), map(memScope))
  void visitCallSPIRVControlBarrier(CallInst *CI) override;

  /// Transform __spirv_Atomic* to atomic_*.
  ///   __spirv_Atomic*(atomic_op, scope, sema, ops, ...) =>
  ///      atomic_*(generic atomic_op, ops, ..., order(sema), map(scope))
  Instruction *visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicIIncrement / OpAtomicIDecrement to
  /// atomic_fetch_add_explicit / atomic_fetch_sub_explicit
  Instruction *visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) override;

  /// Conduct generic mutations for all atomic builtins
  CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) override;

  /// Transform atomic builtin name into correct ocl-dependent name
  Instruction *mutateAtomicName(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicCompareExchange/Weak into
  /// compare_exchange_strong/weak_explicit
  Instruction *visitCallSPIRVAtomicCmpExchg(CallInst *CI, Op OC) override;
};

class SPIRVToOCL20Pass : public llvm::PassInfoMixin<SPIRVToOCL20Pass>,
                         public SPIRVToOCL20Base {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runSPIRVToOCL(M) ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all();
  }
};

class SPIRVToOCL20Legacy : public SPIRVToOCLLegacy, public SPIRVToOCL20Base {
public:
  SPIRVToOCL20Legacy() : SPIRVToOCLLegacy(ID) {
    initializeSPIRVToOCL20LegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  static char ID;
};

char SPIRVToOCL20Legacy::ID = 0;

bool SPIRVToOCL20Legacy::runOnModule(Module &Module) {
  return SPIRVToOCL20Base::runSPIRVToOCL(Module);
}
bool SPIRVToOCL20Base::runSPIRVToOCL(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  visit(*M);

  eraseUselessFunctions(&Module);

  LLVM_DEBUG(dbgs() << "After SPIRVToOCL20:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void SPIRVToOCL20Base::visitCallSPIRVMemoryBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Value *MemScope =
            SPIRV::transSPIRVMemoryScopeIntoOCLMemoryScope(Args[0], CI);
        Value *MemFenceFlags =
            SPIRV::transSPIRVMemorySemanticsIntoOCLMemFenceFlags(Args[1], CI);
        Value *MemOrder =
            SPIRV::transSPIRVMemorySemanticsIntoOCLMemoryOrder(Args[1], CI);

        Args.resize(3);
        Args[0] = MemFenceFlags;
        Args[1] = MemOrder;
        Args[2] = MemScope;

        return kOCLBuiltinName::AtomicWorkItemFence;
      },
      &Attrs);
}

void SPIRVToOCL20Base::visitCallSPIRVControlBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Attrs = Attrs.addAttribute(CI->getContext(), AttributeList::FunctionIndex,
                             Attribute::Convergent);
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        auto GetArg = [=](unsigned I) {
          return cast<ConstantInt>(Args[I])->getZExtValue();
        };
        auto ExecScope = static_cast<Scope>(GetArg(0));
        Value *MemScope =
            getInt32(M, rmap<OCLScopeKind>(static_cast<Scope>(GetArg(1))));
        Value *MemFenceFlags =
            SPIRV::transSPIRVMemorySemanticsIntoOCLMemFenceFlags(Args[2], CI);

        Args.resize(2);
        Args[0] = MemFenceFlags;
        Args[1] = MemScope;

        return (ExecScope == ScopeWorkgroup) ? kOCLBuiltinName::WorkGroupBarrier
                                             : kOCLBuiltinName::SubGroupBarrier;
      },
      &Attrs);
}

Instruction *SPIRVToOCL20Base::mutateAtomicName(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

Instruction *SPIRVToOCL20Base::visitCallSPIRVAtomicBuiltin(CallInst *CI,
                                                           Op OC) {
  CallInst *CIG = mutateCommonAtomicArguments(CI, OC);

  Instruction *NewCI = nullptr;
  switch (OC) {
  case OpAtomicIIncrement:
  case OpAtomicIDecrement:
    NewCI = visitCallSPIRVAtomicIncDec(CIG, OC);
    break;
  case OpAtomicCompareExchange:
  case OpAtomicCompareExchangeWeak:
    NewCI = visitCallSPIRVAtomicCmpExchg(CIG, OC);
    break;
  default:
    NewCI = mutateAtomicName(CIG, OC);
  }

  return NewCI;
}

Instruction *SPIRVToOCL20Base::visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        // Since OpenCL 2.0 doesn't have atomic_inc and atomic_dec builtins,
        // we translate these instructions to atomic_fetch_add_explicit and
        // atomic_fetch_sub_explicit OpenCL 2.0 builtins with "operand" argument
        // = 1.
        auto Name = OCLSPIRVBuiltinMap::rmap(
            OC == OpAtomicIIncrement ? OpAtomicIAdd : OpAtomicISub);
        auto Ptr = findFirstPtr(Args);
        Type *ValueTy =
            cast<PointerType>(Args[Ptr]->getType())->getElementType();
        assert(ValueTy->isIntegerTy());
        Args.insert(Args.begin() + 1, llvm::ConstantInt::get(ValueTy, 1));
        return Name;
      },
      &Attrs);
}

CallInst *SPIRVToOCL20Base::mutateCommonAtomicArguments(CallInst *CI, Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();

  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        for (size_t I = 0; I < Args.size(); ++I) {
          Value *PtrArg = Args[I];
          Type *PtrArgTy = PtrArg->getType();
          if (PtrArgTy->isPointerTy()) {
            if (PtrArgTy->getPointerAddressSpace() != SPIRAS_Generic) {
              Type *FixedPtr = PtrArgTy->getPointerElementType()->getPointerTo(
                  SPIRAS_Generic);
              Args[I] = CastInst::CreatePointerBitCastOrAddrSpaceCast(
                  PtrArg, FixedPtr, PtrArg->getName() + ".as", CI);
            }
          }
        }
        auto Ptr = findFirstPtr(Args);
        auto Name = OCLSPIRVBuiltinMap::rmap(OC);
        auto NumOrder = getSPIRVAtomicBuiltinNumMemoryOrderArgs(OC);
        auto ScopeIdx = Ptr + 1;
        auto OrderIdx = Ptr + 2;

        Args[ScopeIdx] =
            SPIRV::transSPIRVMemoryScopeIntoOCLMemoryScope(Args[ScopeIdx], CI);
        for (size_t I = 0; I < NumOrder; ++I) {
          Args[OrderIdx + I] =
              SPIRV::transSPIRVMemorySemanticsIntoOCLMemoryOrder(
                  Args[OrderIdx + I], CI);
        }
        std::swap(Args[ScopeIdx], Args.back());
        return Name;
      },
      &Attrs);
}

Instruction *SPIRVToOCL20Base::visitCallSPIRVAtomicCmpExchg(CallInst *CI,
                                                            Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Instruction *PInsertBefore = CI;

  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        // OpAtomicCompareExchange[Weak] semantics is different from
        // atomic_compare_exchange_[strong|weak] semantics as well as
        // arguments order.
        // OCL built-ins returns boolean value and stores a new/original
        // value by pointer passed as 2nd argument (aka expected) while SPIR-V
        // instructions returns this new/original value as a resulting value.
        AllocaInst *PExpected = new AllocaInst(CI->getType(), 0, "expected",
                                               &(*PInsertBefore->getParent()
                                                      ->getParent()
                                                      ->getEntryBlock()
                                                      .getFirstInsertionPt()));
        PExpected->setAlignment(
            Align(CI->getType()->getScalarSizeInBits() / 8));
        new StoreInst(Args[1], PExpected, PInsertBefore);
        unsigned AddrSpc = SPIRAS_Generic;
        Type *PtrTyAS =
            PExpected->getType()->getElementType()->getPointerTo(AddrSpc);
        Args[1] = CastInst::CreatePointerBitCastOrAddrSpaceCast(
            PExpected, PtrTyAS, PExpected->getName() + ".as", PInsertBefore);
        std::swap(Args[3], Args[4]);
        std::swap(Args[2], Args[3]);
        RetTy = Type::getInt1Ty(*Ctx);
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      [=](CallInst *CI) -> Instruction * {
        // OCL built-ins atomic_compare_exchange_[strong|weak] return boolean
        // value. So, to obtain the same value as SPIR-V instruction is
        // returning it has to be loaded from the memory where 'expected'
        // value is stored. This memory must contain the needed value after a
        // call to OCL built-in is completed.
        return new LoadInst(
            CI->getArgOperand(1)->getType()->getPointerElementType(),
            CI->getArgOperand(1), "original", PInsertBefore);
      },
      &Attrs);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVToOCL20Legacy, "spvtoocl20",
                "Translate SPIR-V builtins to OCL 2.0 builtins", false, false)

ModulePass *llvm::createSPIRVToOCL20Legacy() {
  return new SPIRVToOCL20Legacy();
}
