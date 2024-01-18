// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/builtin_info.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "transform/passes.h"

using namespace llvm;
using namespace vecz;

PreservedAnalyses BuiltinInliningPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  bool modified = false;
  bool needToRunInliner = false;
  llvm::FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (Function &F : M.functions()) {
    SmallVector<Instruction *, 4> ToDelete;
    for (BasicBlock &BB : F) {
      if (!FAM.getResult<VectorizationUnitAnalysis>(F).hasResult()) {
        continue;
      }
      for (Instruction &I : BB) {
        // Only look at call instructions as those are the only things that can
        // be builtins.
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI) {
          continue;
        }

        bool NeedLLVMInline = false;
        Value *NewCI = processCallSite(CI, NeedLLVMInline);
        needToRunInliner |= NeedLLVMInline;
        if ((NewCI == CI) || !NewCI) {
          continue;
        }

        if (!CI->getType()->isVoidTy()) {
          CI->replaceAllUsesWith(NewCI);
        }
        ToDelete.push_back(CI);
        modified = true;
      }
    }
    // Clean up.
    while (!ToDelete.empty()) {
      Instruction *I = ToDelete.pop_back_val();
      I->eraseFromParent();
    }
  }

  // Run the LLVM inliner if some calls were marked as needing inlining.
  if (needToRunInliner) {
    llvm::legacy::PassManager PM;
    PM.add(llvm::createAlwaysInlinerLegacyPass());
    modified |= PM.run(M);
  }

  // Recursively run the pass to inline any newly introduced functions.
  if (modified) {
    run(M, AM);
  }

  return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

static Value *emitBuiltinMemSet(Function *F, IRBuilder<> &B,
                                ArrayRef<Value *> Args, llvm::CallBase *CB) {
  LLVMContext &Context = F->getContext();
  auto &DL = F->getParent()->getDataLayout();
  const unsigned PtrBits = DL.getPointerSizeInBits();

  // Check the alignment constraints do not exceed the algorithmic requirements
  // of doing 64 bits at time

  // @llvm.memset defines 0 and 1 to both mean no alignment.
  const auto &MSI = cast<MemSetInst>(CB);

  // Note that once LLVM 8.0 is deprecated we can use actual alignment classes
  const Align Alignment = MSI->getDestAlign().valueOrOne();
  const Align Int64Alignment = DL.getABITypeAlign(B.getInt64Ty());
  if (Alignment < std::max(Int64Alignment, Align(8u))) {
    return nullptr;
  }

  Value *DstPtr = Args[0];
  Type *Int8Ty = B.getInt8Ty();

  Value *StoredValue = Args[1];
  const bool IsVolatile = (Args.back() == ConstantInt::getTrue(Context));
  llvm::StoreInst *MS = nullptr;

  // For nicely named IR instructions
  const std::string DstName = DstPtr->getName().str();

  // We can only replace memset instructions if they have a constant length
  ConstantInt *CL = dyn_cast<ConstantInt>(Args[2]);
  if (!CL) {
    return nullptr;
  }
  const int64_t Bytes = CL->getValue().getZExtValue();

  // Unlike memcpy, if we want to use 64bit stores in memset we need to
  // construct the 64bit value from a 8bit one.
  // First, check if we can get the value at compile time
  ConstantInt *ConstantValue = dyn_cast<ConstantInt>(StoredValue);
  Value *StoredValue64 = nullptr;
  if (ConstantValue) {
    // If we can get the value at compile time, calculate the 64bit value at
    // compile time as well.
    const unsigned IntValue = ConstantValue->getZExtValue();
    APInt APValue(64, IntValue);
    for (int i = 1; IntValue && i < 8; ++i) {
      APValue |= APValue << 8;
    }
    StoredValue64 = ConstantInt::get(Context, APValue);
  } else {
    StoredValue64 = B.CreateZExt(StoredValue, Type::getInt64Ty(Context));
    for (int i = 1; i < 8; ++i) {
      StoredValue64 = B.CreateOr(
          StoredValue64,
          B.CreateShl(StoredValue64,
                      llvm::ConstantInt::get(Context, llvm::APInt(64, 8))));
    }
    // If we can't get the value at compile time, we have to emit instructions
    // to generate it at runtime.
  }
  StoredValue64->setName("ms64val");

  // Emit enough loads and stores to replicate the behaviour of memset.
  int64_t byte = 0;
  // Initially we use 64bit loads and stores, in order to avoid emitting too
  // many instructions.
  // We can't just get an Int64PtrTy because we need the correct address space
  Type *DstInt64PtrTy = B.getInt64Ty()->getPointerTo(
      cast<PointerType>(DstPtr->getType())->getAddressSpace());

  for (; byte <= Bytes - 8; byte += 8) {
    Value *Idx = B.getIntN(PtrBits, byte);
    Value *OffsetDstPtr = B.CreateBitCast(
        B.CreateInBoundsGEP(Int8Ty, DstPtr, Idx), DstInt64PtrTy, DstName);
    MS = B.CreateStore(StoredValue64, OffsetDstPtr, IsVolatile);

    // Set alignments for store to be minimum of that from
    // the instruction and what is required for 8 byte stores
    const Align StoreAlign =
        byte == 0 ? Alignment : std::min(Align(8u), Alignment);
    MS->setAlignment(StoreAlign);
  }
  // ...and then we fill in the remaining with 8bit stores.
  for (; byte < Bytes; byte += 1) {
    Value *Idx = B.getIntN(PtrBits, byte);
    Value *OffsetDstPtr = B.CreateInBoundsGEP(Int8Ty, DstPtr, Idx, DstName);
    MS = B.CreateStore(StoredValue, OffsetDstPtr, IsVolatile);
    MS->setAlignment(llvm::Align(1));
  }

  return MS;
}

static Value *emitBuiltinMemCpy(Function *F, IRBuilder<> &B,
                                ArrayRef<Value *> Args, llvm::CallBase *CB) {
  LLVMContext &Context = F->getContext();
  auto &DL = F->getParent()->getDataLayout();

  const auto &MSI = cast<MemCpyInst>(CB);
  const Align DestAlignment = MSI->getDestAlign().valueOrOne();
  const Align SourceAlignment = MSI->getSourceAlign().valueOrOne();
  const Align Int64Alignment = DL.getABITypeAlign(B.getInt64Ty());

  if (DestAlignment < std::max(Int64Alignment, Align(8u))) {
    return nullptr;
  }

  if (SourceAlignment < std::max(Int64Alignment, Align(8u))) {
    return nullptr;
  }

  const unsigned PtrBits = DL.getPointerSizeInBits();

  Value *DstPtr = Args[0];
  Value *SrcPtr = Args[1];
  Type *Int8Ty = B.getInt8Ty();

  const bool IsVolatile = (Args.back() == ConstantInt::getTrue(Context));
  llvm::StoreInst *MC = nullptr;

  // For nicely named IR instructions
  const std::string DstName = DstPtr->getName().str();
  const std::string SrcName = SrcPtr->getName().str();

  // Get the length as a constant
  ConstantInt *CL = dyn_cast<ConstantInt>(Args[2]);
  // We can only replace memcpy instructions if they have a constant length
  if (!CL) {
    return nullptr;
  }
  const int64_t Length = CL->getValue().getSExtValue();

  // Emit enough stores to replicate the behaviour of memcpy.
  int64_t byte = 0;
  // Initially we use 64bit loads and stores, in order to avoid emitting too
  // many instructions...
  // We can't just get an Int64PtrTy because we need the correct address space
  Type *Int64Ty = B.getInt64Ty();
  Type *SrcInt64PtrTy = Int64Ty->getPointerTo(
      cast<PointerType>(SrcPtr->getType())->getAddressSpace());
  Type *DstInt64PtrTy = Int64Ty->getPointerTo(
      cast<PointerType>(DstPtr->getType())->getAddressSpace());

  for (; byte <= Length - 8; byte += 8) {
    Value *Idx = B.getIntN(PtrBits, byte);
    Value *OffsetSrcPtr = B.CreateBitCast(
        B.CreateInBoundsGEP(Int8Ty, SrcPtr, Idx), SrcInt64PtrTy);
    Value *OffsetDstPtr = B.CreateBitCast(
        B.CreateInBoundsGEP(Int8Ty, DstPtr, Idx), DstInt64PtrTy, DstName);
    LoadInst *LoadValue =
        B.CreateLoad(Int64Ty, OffsetSrcPtr, IsVolatile, SrcName);
    MC = B.CreateStore(LoadValue, OffsetDstPtr, IsVolatile);

    // Set alignments for stores and loads to be minimum of that from
    // the instruction and what is required for 8 byte load/stores
    const Align StoreAlign =
        byte == 0 ? DestAlignment : std::min(Align(8u), DestAlignment);
    MC->setAlignment(StoreAlign);
    const Align LoadAlign =
        byte == 0 ? SourceAlignment : std::min(Align(8u), SourceAlignment);
    LoadValue->setAlignment(LoadAlign);
  }
  // ...and then we fill in the remaining with 8bit stores.
  for (; byte < Length; byte += 1) {
    Value *Idx = B.getIntN(PtrBits, byte);
    Value *OffsetSrcPtr = B.CreateInBoundsGEP(Int8Ty, SrcPtr, Idx);
    Value *OffsetDstPtr = B.CreateInBoundsGEP(Int8Ty, DstPtr, Idx, DstName);
    LoadInst *LoadValue =
        B.CreateLoad(Int8Ty, OffsetSrcPtr, IsVolatile, SrcName);
    MC = B.CreateStore(LoadValue, OffsetDstPtr, IsVolatile);
    LoadValue->setAlignment(llvm::Align(1));
    MC->setAlignment(llvm::Align(1));
  }

  return MC;
}

Value *BuiltinInliningPass::processCallSite(CallInst *CI,
                                            bool &NeedLLVMInline) {
  NeedLLVMInline = false;

  Function *Callee = CI->getCalledFunction();
  if (!Callee) {
    return CI;
  }

  // Mark user function as needing inlining by LLVM, unless it has the NoInline
  // attribute
  if (!Callee->isDeclaration() &&
      !Callee->hasFnAttribute(Attribute::NoInline)) {
    CI->addFnAttr(Attribute::AlwaysInline);
    NeedLLVMInline = true;
    return CI;
  }

  // Specially inline some LLVM intrinsics.
  if (Callee->isIntrinsic()) {
    if (Callee->getIntrinsicID() == Intrinsic::memcpy) {
      IRBuilder<> B(CI);
      const SmallVector<Value *, 4> Args(CI->args());
      if (Value *Impl = emitBuiltinMemCpy(Callee, B, Args, CI)) {
        return Impl;
      }
    }

    if (Callee->getIntrinsicID() == Intrinsic::memset) {
      IRBuilder<> B(CI);
      const SmallVector<Value *, 4> Args(CI->args());
      if (Value *Impl = emitBuiltinMemSet(Callee, B, Args, CI)) {
        return Impl;
      }
    }
  }

  return CI;
}
