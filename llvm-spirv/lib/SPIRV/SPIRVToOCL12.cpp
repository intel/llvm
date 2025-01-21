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

char SPIRVToOCL12Legacy::ID = 0;

bool SPIRVToOCL12Legacy::runOnModule(Module &Module) {
  return SPIRVToOCL12Base::runSPIRVToOCL(Module);
}

bool SPIRVToOCL12Base::runSPIRVToOCL(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  // Lower builtin variables to builtin calls first.
  lowerBuiltinVariablesToCalls(M);
  translateOpaqueTypes();

  visit(*M);

  postProcessBuiltinsReturningStruct(M);
  postProcessBuiltinsWithArrayArguments(M);

  eraseUselessFunctions(&Module);

  LLVM_DEBUG(dbgs() << "After SPIRVToOCL12:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void SPIRVToOCL12Base::visitCallSPIRVMemoryBarrier(CallInst *CI) {
  mutateCallInst(CI, kOCLBuiltinName::MemFence)
      .mapArg(1,
              [=](Value *V) {
                return transSPIRVMemorySemanticsIntoOCLMemFenceFlags(V, CI);
              })
      .removeArg(0);
}

void SPIRVToOCL12Base::visitCallSPIRVControlBarrier(CallInst *CI) {
  mutateCallInst(CI, kOCLBuiltinName::Barrier)
      .mapArg(2,
              [=](Value *V) {
                return transSPIRVMemorySemanticsIntoOCLMemFenceFlags(V, CI);
              })
      .removeArg(1)
      .removeArg(0);
}

void SPIRVToOCL12Base::visitCallSPIRVSplitBarrierINTEL(CallInst *CI, Op OC) {
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC))
      .mapArg(2,
              [=](Value *V) {
                return transSPIRVMemorySemanticsIntoOCLMemFenceFlags(V, CI);
              })
      .removeArg(1)
      .removeArg(0);
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) {
  mutateCallInst(CI, mapAtomicName(OC, CI->getType()))
      .removeArg(2)
      .removeArg(1);
}

CallInst *SPIRVToOCL12Base::mutateCommonAtomicArguments(CallInst *CI, Op OC) {
  auto Ptr = findFirstPtr(CI->args());
  auto NumOrder = getSPIRVAtomicBuiltinNumMemoryOrderArgs(OC);
  auto ArgsToRemove = NumOrder + 1; // OpenCL1.2 builtins does not use
                                    // scope and memory order arguments

  auto Mutator = mutateCallInst(CI, mapAtomicName(OC, CI->getType()));
  Mutator.removeArgs(Ptr + 1, ArgsToRemove);
  return cast<CallInst>(Mutator.getMutated());
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicUMinUMax(CallInst *CI, Op OC) {
  mutateCallInst(CI, mapAtomicName(OC, CI->getType()))
      .moveArg(3, 1)
      .removeArg(3)
      .removeArg(2);
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicLoad(CallInst *CI) {
  // There is no atomic_load in OpenCL 1.2 spec.
  // Emit this builtin via call of atomic_add(*p, 0).
  Type *PtrElemTy = CI->getType();
  mutateCallInst(CI, mapAtomicName(OpAtomicIAdd, PtrElemTy))
      .removeArg(2)
      .removeArg(1)
      .appendArg(Constant::getNullValue(PtrElemTy));
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicStore(CallInst *CI) {
  Type *RetTy = CI->getArgOperand(3)->getType();
  mutateCallInst(CI, mapAtomicName(OpAtomicExchange, RetTy))
      .removeArg(2)
      .removeArg(1)
      .changeReturnType(RetTy, nullptr);
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicFlagClear(CallInst *CI) {
  Type *RetTy = Type::getInt32Ty(M->getContext());
  mutateCallInst(CI, mapAtomicName(OpAtomicExchange, RetTy))
      .removeArg(2)
      .removeArg(1)
      .appendArg(getInt32(M, 0))
      .changeReturnType(RetTy, nullptr);
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicFlagTestAndSet(CallInst *CI) {
  Type *RetTy = Type::getInt32Ty(M->getContext());
  mutateCallInst(CI, mapAtomicName(OpAtomicExchange, RetTy))
      .removeArg(2)
      .removeArg(1)
      .appendArg(getInt32(M, 1))
      .changeReturnType(RetTy, [](IRBuilder<> &Builder, CallInst *NewCI) {
        return Builder.CreateTrunc(NewCI, Builder.getInt1Ty());
      });
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicCmpExchg(CallInst *CI) {
  mutateCallInst(CI, mapAtomicName(OpAtomicCompareExchange, CI->getType()))
      .removeArg(3)
      .removeArg(2)
      .removeArg(1)
      // SPIRV OpAtomicCompareExchange and OpAtomicCompareExchangeWeak has Value
      // and Comparator in different order than ocl functions both of them are
      // translated into atomic_cmpxchg
      .moveArg(2, 1);
}

void SPIRVToOCL12Base::visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) {
  switch (OC) {
  case OpAtomicLoad:
    visitCallSPIRVAtomicLoad(CI);
    break;
  case OpAtomicStore:
    visitCallSPIRVAtomicStore(CI);
    break;
  case OpAtomicFlagClear:
    visitCallSPIRVAtomicFlagClear(CI);
    break;
  case OpAtomicFlagTestAndSet:
    visitCallSPIRVAtomicFlagTestAndSet(CI);
    break;
  case OpAtomicUMin:
  case OpAtomicUMax:
    visitCallSPIRVAtomicUMinUMax(CI, OC);
    break;
  case OpAtomicCompareExchange:
  case OpAtomicCompareExchangeWeak:
    visitCallSPIRVAtomicCmpExchg(CI);
    break;
  default:
    mutateCommonAtomicArguments(CI, OC);
  }
}

void SPIRVToOCL12Base::visitCallSPIRVEnqueueKernel(CallInst *CI, Op OC) {
  assert(0 && "OpenCL 1.2 doesn't support enqueue_kernel!");
}

std::string SPIRVToOCL12Base::mapFPAtomicName(Op OC) {
  assert(isFPAtomicOpCode(OC) && "Not intended to handle other opcodes than "
                                 "AtomicF{Add/Min/Max}EXT!");
  switch (OC) {
  case OpAtomicFAddEXT:
    return "atomic_add";
  case OpAtomicFMinEXT:
    return "atomic_min";
  case OpAtomicFMaxEXT:
    return "atomic_max";
  default:
    llvm_unreachable("Unsupported opcode!");
  }
}

void SPIRVToOCL12Base::mutateAtomicName(CallInst *CI, Op OC) {
  mutateCallInst(CI, OCL12SPIRVBuiltinMap::rmap(OC));
}

std::string SPIRVToOCL12Base::mapAtomicName(Op OC, Type *Ty) {
  std::string Prefix = Ty->isIntegerTy(64) ? kOCLBuiltinName::AtomPrefix
                                           : kOCLBuiltinName::AtomicPrefix;
  // Map fp atomic instructions to regular OpenCL built-ins.
  if (isFPAtomicOpCode(OC))
    return mapFPAtomicName(OC);
  return Prefix += OCL12SPIRVBuiltinMap::rmap(OC);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVToOCL12Legacy, "spvtoocl12",
                "Translate SPIR-V builtins to OCL 1.2 builtins", false, false)

ModulePass *llvm::createSPIRVToOCL12Legacy() {
  return new SPIRVToOCL12Legacy();
}
