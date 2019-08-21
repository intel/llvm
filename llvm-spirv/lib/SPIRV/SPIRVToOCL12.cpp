//===- SPIRVToOCL12.cpp - Transform SPIR-V builtins to OCL 1.2
// builtins------===//
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
// This file implements transform of SPIR-V builtins to OCL 1.2 builtins.
//
//===----------------------------------------------------------------------===//
#include "SPIRVToOCL.h"
#include "llvm/IR/Verifier.h"

#define DEBUG_TYPE "spvtocl12"

namespace SPIRV {

class SPIRVToOCL12 : public SPIRVToOCL {
public:
  SPIRVToOCL12() {
    initializeSPIRVToOCL12Pass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;

  /// Transform __spirv_MemoryBarrier to atomic_work_item_fence.
  ///   __spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  void visitCallSPIRVMemoryBarrier(CallInst *CI) override;

  /// Transform __spirv_ControlBarrier to barrier.
  ///   __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///       barrier(flag(sema))
  void visitCallSPIRVControlBarrier(CallInst *CI) override;

  /// Transform __spirv_OpAtomic functions. It firstly conduct generic
  /// mutations for all builtins and then mutate some of them seperately
  Instruction *visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicIIncrement / OpAtomicIDecrement to
  /// atomic_inc / atomic_dec
  Instruction *visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicUMin/SMin/UMax/SMax into
  /// atomic_min/atomic_max, as there is no distinction in OpenCL 1.2
  /// between signed and unsigned version of those functions
  Instruction *visitCallSPIRVAtomicUMinUMax(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicLoad to atomic_add(*ptr, 0)
  Instruction *visitCallSPIRVAtomicLoad(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicStore to atomic_xchg(*ptr, value)
  Instruction *visitCallSPIRVAtomicStore(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicFlagClear to atomic_xchg(*ptr, 0)
  /// with ignoring the result
  Instruction *visitCallSPIRVAtomicFlagClear(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicFlagTestAndTest to
  /// (bool)atomic_xchg(*ptr, 1)
  Instruction *visitCallSPIRVAtomicFlagTestAndSet(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicCompareExchange and
  /// __spirv_OpAtomicCompareExchangeWeak into atomic_cmpxchg. There is no
  /// weak version of function in OpenCL 1.2
  Instruction *visitCallSPIRVAtomicCmpExchg(CallInst *CI, Op OC) override;

  /// Conduct generic mutations for all atomic builtins
  CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) override;

  /// Transform atomic builtin name into correct ocl-dependent name
  Instruction *mutateAtomicName(CallInst *CI, Op OC) override;
};

bool SPIRVToOCL12::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  visit(*M);

  translateMangledAtomicTypeName();

  eraseUselessFunctions(&Module);

  LLVM_DEBUG(dbgs() << "After SPIRVToOCL12:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void SPIRVToOCL12::visitCallSPIRVMemoryBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        if (auto Arg = dyn_cast<ConstantInt>(Args[1])) {
          auto Sema = mapSPIRVMemSemanticToOCL(Arg->getZExtValue());
          Args.resize(1);
          Args[0] = getInt32(M, Sema.first);
        } else {
          CallInst *TransCall = dyn_cast<CallInst>(Args[1]);
          Function *F = TransCall ? TransCall->getCalledFunction() : nullptr;
          if (F && F->getName().equals(kSPIRVName::TranslateOCLMemScope)) {
            Args[0] = TransCall->getArgOperand(0);
          } else {
            int ClMemFenceMask = MemorySemanticsWorkgroupMemoryMask |
                                 MemorySemanticsCrossWorkgroupMemoryMask |
                                 MemorySemanticsImageMemoryMask;
            Args[0] = getOrCreateSwitchFunc(
                kSPIRVName::TranslateSPIRVMemFence, Args[1],
                OCLMemFenceExtendedMap::getRMap(), true /*IsReverse*/, None, CI,
                M, ClMemFenceMask);
          }
          Args.resize(1);
        }
        return kOCLBuiltinName::MemFence;
      },
      &Attrs);
}

void SPIRVToOCL12::visitCallSPIRVControlBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Attrs = Attrs.addAttribute(CI->getContext(), AttributeList::FunctionIndex,
                             Attribute::Convergent);
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        if (auto Arg = dyn_cast<ConstantInt>(Args[2])) {
          auto Sema = mapSPIRVMemSemanticToOCL(Arg->getZExtValue());
          Args.resize(1);
          Args[0] = getInt32(M, Sema.first);
        } else {
          CallInst *TransCall = dyn_cast<CallInst>(Args[2]);
          Function *F = TransCall ? TransCall->getCalledFunction() : nullptr;
          if (F && F->getName().equals(kSPIRVName::TranslateOCLMemScope)) {
            Args[0] = TransCall->getArgOperand(0);
          } else {
            int ClMemFenceMask = MemorySemanticsWorkgroupMemoryMask |
                                 MemorySemanticsCrossWorkgroupMemoryMask |
                                 MemorySemanticsImageMemoryMask;
            Args[0] = getOrCreateSwitchFunc(
                kSPIRVName::TranslateSPIRVMemFence, Args[2],
                OCLMemFenceExtendedMap::getRMap(), true /*IsReverse*/, None, CI,
                M, ClMemFenceMask);
          }
          Args.resize(1);
        }
        return kOCLBuiltinName::Barrier;
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(1);
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

CallInst *SPIRVToOCL12::mutateCommonAtomicArguments(CallInst *CI, Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();

  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        auto Ptr = findFirstPtr(Args);
        auto NumOrder = getSPIRVAtomicBuiltinNumMemoryOrderArgs(OC);
        auto ArgsToRemove = NumOrder + 1; // OpenCL1.2 builtins does not use
                                          // scope and memory order arguments
        auto StartIdx = Ptr + 1;
        auto StopIdx = StartIdx + ArgsToRemove;
        Args.erase(Args.begin() + StartIdx, Args.begin() + StopIdx);
        return OCL12SPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicUMinUMax(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        std::swap(Args[1], Args[3]);
        Args.resize(2);
        return OCL12SPIRVBuiltinMap::rmap(OC == OpAtomicUMin ? OpAtomicSMin
                                                             : OpAtomicSMax);
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicLoad(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(1);
        // There is no atomic_load in OpenCL 1.2 spec.
        // Emit this builtin via call of atomic_add(*p, 0).
        Type *ptrElemTy = Args[0]->getType()->getPointerElementType();
        Args.push_back(Constant::getNullValue(ptrElemTy));
        return OCL12SPIRVBuiltinMap::rmap(OpAtomicIAdd);
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicStore(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        std::swap(Args[1], Args[3]);
        Args.resize(2);
        // The type of the value pointed to by Pointer (1st argument)
        // must be the same as Result Type.
        RetTy = Args[0]->getType()->getPointerElementType();
        return OCL12SPIRVBuiltinMap::rmap(OpAtomicExchange);
      },
      [=](CallInst *CI) -> Instruction * { return CI; }, &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicFlagClear(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        Args.resize(1);
        Args.push_back(getInt32(M, 0));
        RetTy = Type::getInt32Ty(M->getContext());
        return OCL12SPIRVBuiltinMap::rmap(OpAtomicExchange);
      },
      [=](CallInst *CI) -> Instruction * { return CI; }, &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicFlagTestAndSet(CallInst *CI,
                                                              Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        Args.resize(1);
        Args.push_back(getInt32(M, 1));
        RetTy = Type::getInt32Ty(M->getContext());
        return OCL12SPIRVBuiltinMap::rmap(OpAtomicExchange);
      },
      [=](CallInst *CI) -> Instruction * {
        return BitCastInst::Create(Instruction::Trunc, CI,
                                   Type::getInt1Ty(CI->getContext()), "",
                                   CI->getNextNode());
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicCmpExchg(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.erase(Args.begin() + 1, Args.begin() + 4);
        // SPIRV OpAtomicCompareExchange and OpAtomicCompareExchangeWeak
        // has Value and Comparator in different order than ocl functions
        // both of them are translated into atomic_cmpxchg
        std::swap(Args[1], Args[2]);
        return OCL12SPIRVBuiltinMap::rmap(OpAtomicCompareExchange);
      },
      &Attrs);
}

Instruction *SPIRVToOCL12::visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) {
  Instruction *NewCI = nullptr;
  switch (OC) {
  case OpAtomicLoad:
    NewCI = visitCallSPIRVAtomicLoad(CI, OC);
    break;
  case OpAtomicStore:
    NewCI = visitCallSPIRVAtomicStore(CI, OC);
    break;
  case OpAtomicFlagClear:
    NewCI = visitCallSPIRVAtomicFlagClear(CI, OC);
    break;
  case OpAtomicFlagTestAndSet:
    NewCI = visitCallSPIRVAtomicFlagTestAndSet(CI, OC);
    break;
  case OpAtomicUMin:
  case OpAtomicUMax:
    NewCI = visitCallSPIRVAtomicUMinUMax(CI, OC);
    break;
  case OpAtomicCompareExchange:
  case OpAtomicCompareExchangeWeak:
    NewCI = visitCallSPIRVAtomicCmpExchg(CI, OC);
    break;
  default:
    NewCI = mutateCommonAtomicArguments(CI, OC);
  }

  return NewCI;
}

Instruction *SPIRVToOCL12::mutateAtomicName(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return OCL12SPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVToOCL12, "spvtoocl12",
                "Translate SPIR-V builtins to OCL 1.2 builtins", false, false)

ModulePass *llvm::createSPIRVToOCL12() { return new SPIRVToOCL12(); }
