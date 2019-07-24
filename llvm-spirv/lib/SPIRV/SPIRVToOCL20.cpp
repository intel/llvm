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

#include "SPIRVToOCL.h"
#include "llvm/IR/Verifier.h"

namespace SPIRV {

class SPIRVToOCL20 : public SPIRVToOCL {
public:
  SPIRVToOCL20() {
    initializeSPIRVToOCL20Pass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;

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
};

bool SPIRVToOCL20::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  visit(*M);

  translateMangledAtomicTypeName();

  eraseUselessFunctions(&Module);

  LLVM_DEBUG(dbgs() << "After SPIRVToOCL20:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void SPIRVToOCL20::visitCallSPIRVMemoryBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI,
                    [=](CallInst *, std::vector<Value *> &Args) {
                      auto GetArg = [=](unsigned I) {
                        return cast<ConstantInt>(Args[I])->getZExtValue();
                      };
                      auto MScope = static_cast<Scope>(GetArg(0));
                      auto Sema = mapSPIRVMemSemanticToOCL(GetArg(1));
                      Args.resize(3);
                      Args[0] = getInt32(M, Sema.first);
                      Args[1] = getInt32(M, Sema.second);
                      Args[2] = getInt32(M, rmap<OCLScopeKind>(MScope));
                      return kOCLBuiltinName::AtomicWorkItemFence;
                    },
                    &Attrs);
}

void SPIRVToOCL20::visitCallSPIRVControlBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Attrs = Attrs.addAttribute(CI->getContext(), AttributeList::FunctionIndex,
                             Attribute::NoDuplicate);
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        auto GetArg = [=](unsigned I) {
          return cast<ConstantInt>(Args[I])->getZExtValue();
        };
        auto ExecScope = static_cast<Scope>(GetArg(0));
        auto MScope = static_cast<Scope>(GetArg(1));
        auto Sema = mapSPIRVMemSemanticToOCL(GetArg(2));
        Args.resize(2);
        Args[0] = getInt32(M, Sema.first);
        Args[1] = getInt32(M, rmap<OCLScopeKind>(MScope));
        return (ExecScope == ScopeWorkgroup) ? kOCLBuiltinName::WorkGroupBarrier
                                             : kOCLBuiltinName::SubGroupBarrier;
      },
      &Attrs);
}

Instruction *SPIRVToOCL20::visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  CallInst *CiFixedAs = mutateCallInstOCL(
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
        return CI->getName();
      },
      &Attrs);
  return SPIRVToOCL::visitCallSPIRVAtomicBuiltin(CiFixedAs, OC);
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVToOCL20, "spvtoocl20",
                "Translate SPIR-V builtins to OCL 2.0 builtins", false, false)

ModulePass *llvm::createSPIRVToOCL20() { return new SPIRVToOCL20(); }
