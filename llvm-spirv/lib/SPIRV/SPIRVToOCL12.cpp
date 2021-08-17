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
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Value *MemFenceFlags =
            transSPIRVMemorySemanticsIntoOCLMemFenceFlags(Args[1], CI);
        Args.assign(1, MemFenceFlags);
        return kOCLBuiltinName::MemFence;
      },
      &Attrs);
}

void SPIRVToOCL12Base::visitCallSPIRVControlBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Attrs = Attrs.addAttribute(CI->getContext(), AttributeList::FunctionIndex,
                             Attribute::Convergent);
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        auto *MemFenceFlags =
            transSPIRVMemorySemanticsIntoOCLMemFenceFlags(Args[2], CI);
        Args.assign(1, MemFenceFlags);
        return kOCLBuiltinName::Barrier;
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(1);
        return mapAtomicName(OC, CI->getType());
      },
      &Attrs);
}

CallInst *SPIRVToOCL12Base::mutateCommonAtomicArguments(CallInst *CI, Op OC) {
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
        return mapAtomicName(OC, CI->getType());
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicUMinUMax(CallInst *CI,
                                                            Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        std::swap(Args[1], Args[3]);
        Args.resize(2);
        return mapAtomicName(OC, CI->getType());
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicLoad(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(1);
        // There is no atomic_load in OpenCL 1.2 spec.
        // Emit this builtin via call of atomic_add(*p, 0).
        Type *ptrElemTy = Args[0]->getType()->getPointerElementType();
        Args.push_back(Constant::getNullValue(ptrElemTy));
        return mapAtomicName(OpAtomicIAdd, ptrElemTy);
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicStore(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        std::swap(Args[1], Args[3]);
        Args.resize(2);
        // The type of the value pointed to by Pointer (1st argument)
        // must be the same as Result Type.
        RetTy = Args[0]->getType()->getPointerElementType();
        return mapAtomicName(OpAtomicExchange, RetTy);
      },
      [=](CallInst *CI) -> Instruction * { return CI; }, &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicFlagClear(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        Args.resize(1);
        Args.push_back(getInt32(M, 0));
        RetTy = Type::getInt32Ty(M->getContext());
        return mapAtomicName(OpAtomicExchange, RetTy);
      },
      [=](CallInst *CI) -> Instruction * { return CI; }, &Attrs);
}

Instruction *
SPIRVToOCL12Base::visitCallSPIRVAtomicFlagTestAndSet(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        Args.resize(1);
        Args.push_back(getInt32(M, 1));
        RetTy = Type::getInt32Ty(M->getContext());
        return mapAtomicName(OpAtomicExchange, RetTy);
      },
      [=](CallInst *CI) -> Instruction * {
        return BitCastInst::Create(Instruction::Trunc, CI,
                                   Type::getInt1Ty(CI->getContext()), "",
                                   CI->getNextNode());
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicCmpExchg(CallInst *CI,
                                                            Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.erase(Args.begin() + 1, Args.begin() + 4);
        // SPIRV OpAtomicCompareExchange and
        // OpAtomicCompareExchangeWeak has Value and
        // Comparator in different order than ocl functions
        // both of them are translated into atomic_cmpxchg
        std::swap(Args[1], Args[2]);
        // Type of return value, pointee of the pointer
        // operand, other operands, all match, and should
        // be integer scalar types.
        return mapAtomicName(OpAtomicCompareExchange, CI->getType());
      },
      &Attrs);
}

Instruction *SPIRVToOCL12Base::visitCallSPIRVAtomicBuiltin(CallInst *CI,
                                                           Op OC) {
  Instruction *NewCI = nullptr;
  switch (OC) {
  case OpAtomicLoad:
    NewCI = visitCallSPIRVAtomicLoad(CI);
    break;
  case OpAtomicStore:
    NewCI = visitCallSPIRVAtomicStore(CI);
    break;
  case OpAtomicFlagClear:
    NewCI = visitCallSPIRVAtomicFlagClear(CI);
    break;
  case OpAtomicFlagTestAndSet:
    NewCI = visitCallSPIRVAtomicFlagTestAndSet(CI);
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

Instruction *SPIRVToOCL12Base::mutateAtomicName(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  return mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return OCL12SPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
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
