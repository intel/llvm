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

#include "SPIRVRegularizeLLVM.h"
#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/ADT/StringExtras.h" // llvm::isDigit
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h" // expandMemSetAsLoop()

#include <set>
#include <vector>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

static bool SPIRVDbgSaveRegularizedModule = false;
static std::string RegularizedModuleTmpFile = "regularized.bc";

char SPIRVRegularizeLLVMLegacy::ID = 0;

bool SPIRVRegularizeLLVMLegacy::runOnModule(Module &Module) {
  return runRegularizeLLVM(Module);
}

std::string SPIRVRegularizeLLVMBase::lowerLLVMIntrinsicName(IntrinsicInst *II) {
  Function *IntrinsicFunc = II->getCalledFunction();
  assert(IntrinsicFunc && "Missing function");
  std::string FuncName = IntrinsicFunc->getName().str();
  std::replace(FuncName.begin(), FuncName.end(), '.', '_');
  FuncName = "spirv." + FuncName;
  return FuncName;
}

void SPIRVRegularizeLLVMBase::lowerIntrinsicToFunction(
    IntrinsicInst *Intrinsic) {
  // For @llvm.memset.* intrinsic cases with constant value and length arguments
  // are emulated via "storing" a constant array to the destination. For other
  // cases we wrap the intrinsic in @spirv.llvm_memset_* function and expand the
  // intrinsic to a loop via expandMemSetAsLoop() from
  // llvm/Transforms/Utils/LowerMemIntrinsics.h
  if (auto *MSI = dyn_cast<MemSetInst>(Intrinsic))
    if (isa<Constant>(MSI->getValue()) && isa<ConstantInt>(MSI->getLength()))
      return; // To be handled in LLVMToSPIRV::transIntrinsicInst

  std::string FuncName = lowerLLVMIntrinsicName(Intrinsic);
  if (Intrinsic->isVolatile())
    FuncName += ".volatile";
  // Redirect @llvm.intrinsic.* call to @spirv.llvm_intrinsic_*
  Function *F = M->getFunction(FuncName);
  if (F) {
    // This function is already linked in.
    Intrinsic->setCalledFunction(F);
    return;
  }
  // TODO copy arguments attributes: nocapture writeonly.
  FunctionCallee FC =
      M->getOrInsertFunction(FuncName, Intrinsic->getFunctionType());
  auto IntrinsicID = Intrinsic->getIntrinsicID();
  Intrinsic->setCalledFunction(FC);

  F = dyn_cast<Function>(FC.getCallee());
  assert(F && "must be a function!");

  switch (IntrinsicID) {
  case Intrinsic::memset: {
    auto *MSI = static_cast<MemSetInst *>(Intrinsic);
    Argument *Dest = F->getArg(0);
    Argument *Val = F->getArg(1);
    Argument *Len = F->getArg(2);
    Argument *IsVolatile = F->getArg(3);
    Dest->setName("dest");
    Val->setName("val");
    Len->setName("len");
    IsVolatile->setName("isvolatile");
    IsVolatile->addAttr(Attribute::ImmArg);
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *MemSet = IRB.CreateMemSet(Dest, Val, Len, MSI->getDestAlign(),
                                    MSI->isVolatile());
    IRB.CreateRetVoid();
    expandMemSetAsLoop(cast<MemSetInst>(MemSet));
    MemSet->eraseFromParent();
    break;
  }
  case Intrinsic::bswap: {
    BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", F);
    IRBuilder<> IRB(EntryBB);
    auto *BSwap = IRB.CreateIntrinsic(Intrinsic::bswap, Intrinsic->getType(),
                                      F->getArg(0));
    IRB.CreateRet(BSwap);
    IntrinsicLowering IL(M->getDataLayout());
    IL.LowerIntrinsicCall(BSwap);
    break;
  }
  default:
    break; // do nothing
  }

  return;
}

void SPIRVRegularizeLLVMBase::lowerFunnelShift(IntrinsicInst *FSHIntrinsic) {
  // Get a separate function - otherwise, we'd have to rework the CFG of the
  // current one. Then simply replace the intrinsic uses with a call to the new
  // function.
  // Expected LLVM IR for the function: i* @spirv.llvm_fsh?_i* (i* %a, i* %b, i*
  // %c)
  FunctionType *FSHFuncTy = FSHIntrinsic->getFunctionType();
  Type *FSHRetTy = FSHFuncTy->getReturnType();
  const std::string FuncName = lowerLLVMIntrinsicName(FSHIntrinsic);
  Function *FSHFunc =
      getOrCreateFunction(M, FSHRetTy, FSHFuncTy->params(), FuncName);

  if (!FSHFunc->empty()) {
    FSHIntrinsic->setCalledFunction(FSHFunc);
    return;
  }
  auto *RotateBB = BasicBlock::Create(M->getContext(), "rotate", FSHFunc);
  IRBuilder<> Builder(RotateBB);
  Type *Ty = FSHFunc->getReturnType();
  // Build the actual funnel shift rotate logic.
  // In the comments, "int" is used interchangeably with "vector of int
  // elements".
  FixedVectorType *VectorTy = dyn_cast<FixedVectorType>(Ty);
  Type *IntTy = VectorTy ? VectorTy->getElementType() : Ty;
  unsigned BitWidth = IntTy->getIntegerBitWidth();
  ConstantInt *BitWidthConstant = Builder.getInt({BitWidth, BitWidth});
  Value *BitWidthForInsts =
      VectorTy ? Builder.CreateVectorSplat(VectorTy->getNumElements(),
                                           BitWidthConstant)
               : BitWidthConstant;
  auto *RotateModVal =
      Builder.CreateURem(/*Rotate*/ FSHFunc->getArg(2), BitWidthForInsts);
  Value *FirstShift = nullptr, *SecShift = nullptr;
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr)
    // Shift the less significant number right, the "rotate" number of bits
    // will be 0-filled on the left as a result of this regular shift.
    FirstShift = Builder.CreateLShr(FSHFunc->getArg(1), RotateModVal);
  else
    // Shift the more significant number left, the "rotate" number of bits
    // will be 0-filled on the right as a result of this regular shift.
    FirstShift = Builder.CreateShl(FSHFunc->getArg(0), RotateModVal);

  // We want the "rotate" number of the more significant int's LSBs (MSBs) to
  // occupy the leftmost (rightmost) "0 space" left by the previous operation.
  // Therefore, subtract the "rotate" number from the integer bitsize...
  auto *SubRotateVal = Builder.CreateSub(BitWidthForInsts, RotateModVal);
  if (FSHIntrinsic->getIntrinsicID() == Intrinsic::fshr)
    // ...and left-shift the more significant int by this number, zero-filling
    // the LSBs.
    SecShift = Builder.CreateShl(FSHFunc->getArg(0), SubRotateVal);
  else
    // ...and right-shift the less significant int by this number, zero-filling
    // the MSBs.
    SecShift = Builder.CreateLShr(FSHFunc->getArg(1), SubRotateVal);

  // A simple binary addition of the shifted ints yields the final result.
  auto *FunnelShiftRes = Builder.CreateOr(FirstShift, SecShift);
  Builder.CreateRet(FunnelShiftRes);

  FSHIntrinsic->setCalledFunction(FSHFunc);
}

void SPIRVRegularizeLLVMBase::buildUMulWithOverflowFunc(Function *UMulFunc) {
  if (!UMulFunc->empty())
    return;

  BasicBlock *EntryBB = BasicBlock::Create(M->getContext(), "entry", UMulFunc);
  IRBuilder<> Builder(EntryBB);
  // Build the actual unsigned multiplication logic with the overflow
  // indication.
  auto *FirstArg = UMulFunc->getArg(0);
  auto *SecondArg = UMulFunc->getArg(1);

  // Do unsigned multiplication Mul = A * B.
  // Then check if unsigned division Div = Mul / A is not equal to B.
  // If so, then overflow has happened.
  auto *Mul = Builder.CreateNUWMul(FirstArg, SecondArg);
  auto *Div = Builder.CreateUDiv(Mul, FirstArg);
  auto *Overflow = Builder.CreateICmpNE(FirstArg, Div);

  // umul.with.overflow intrinsic return a structure, where the first element
  // is the multiplication result, and the second is an overflow bit.
  auto *StructTy = UMulFunc->getReturnType();
  auto *Agg = Builder.CreateInsertValue(UndefValue::get(StructTy), Mul, {0});
  auto *Res = Builder.CreateInsertValue(Agg, Overflow, {1});
  Builder.CreateRet(Res);
}

void SPIRVRegularizeLLVMBase::lowerUMulWithOverflow(
    IntrinsicInst *UMulIntrinsic) {
  // Get a separate function - otherwise, we'd have to rework the CFG of the
  // current one. Then simply replace the intrinsic uses with a call to the new
  // function.
  FunctionType *UMulFuncTy = UMulIntrinsic->getFunctionType();
  Type *FSHLRetTy = UMulFuncTy->getReturnType();
  const std::string FuncName = lowerLLVMIntrinsicName(UMulIntrinsic);
  Function *UMulFunc =
      getOrCreateFunction(M, FSHLRetTy, UMulFuncTy->params(), FuncName);
  buildUMulWithOverflowFunc(UMulFunc);
  UMulIntrinsic->setCalledFunction(UMulFunc);
}

void SPIRVRegularizeLLVMBase::expandVEDWithSYCLTypeSRetArg(Function *F) {
  auto Attrs = F->getAttributes();
  StructType *SRetTy = cast<StructType>(Attrs.getParamStructRetType(0));
  Attrs = Attrs.removeParamAttribute(F->getContext(), 0, Attribute::StructRet);
  std::string Name = F->getName().str();
  CallInst *OldCall = nullptr;
  mutateFunction(
      F,
      [=, &OldCall](CallInst *CI, std::vector<Value *> &Args, Type *&RetTy) {
        Args.erase(Args.begin());
        RetTy = SRetTy->getElementType(0);
        OldCall = CI;
        return Name;
      },
      [=, &OldCall](CallInst *NewCI) {
        IRBuilder<> Builder(OldCall);
        Value *Target =
            Builder.CreateStructGEP(SRetTy, OldCall->getOperand(0), 0);
        return Builder.CreateStore(NewCI, Target);
      },
      nullptr, &Attrs, true);
}

void SPIRVRegularizeLLVMBase::expandVIDWithSYCLTypeByValComp(Function *F) {
  auto Attrs = F->getAttributes();
  auto *CompPtrTy = cast<StructType>(Attrs.getParamByValType(1));
  Attrs = Attrs.removeParamAttribute(F->getContext(), 1, Attribute::ByVal);
  std::string Name = F->getName().str();
  mutateFunction(
      F,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        Type *HalfTy = CompPtrTy->getElementType(0);
        IRBuilder<> Builder(CI);
        auto *Target = Builder.CreateStructGEP(CompPtrTy, CI->getOperand(1), 0);
        Args[1] = Builder.CreateLoad(HalfTy, Target);
        return Name;
      },
      nullptr, &Attrs, true);
}

void SPIRVRegularizeLLVMBase::expandSYCLTypeUsing(Module *M) {
  std::vector<Function *> ToExpandVEDWithSYCLTypeSRetArg;
  std::vector<Function *> ToExpandVIDWithSYCLTypeByValComp;

  for (auto &F : *M) {
    if (F.getName().startswith("_Z28__spirv_VectorExtractDynamic") &&
        F.hasStructRetAttr()) {
      auto *SRetTy = F.getParamStructRetType(0);
      if (isSYCLHalfType(SRetTy) || isSYCLBfloat16Type(SRetTy))
        ToExpandVEDWithSYCLTypeSRetArg.push_back(&F);
      else
        llvm_unreachable("The return type of the VectorExtractDynamic "
                         "instruction cannot be a structure other than SYCL "
                         "half.");
    }
    if (F.getName().startswith("_Z27__spirv_VectorInsertDynamic") &&
        F.getArg(1)->getType()->isPointerTy()) {
      auto *ET = F.getParamByValType(1);
      if (isSYCLHalfType(ET) || isSYCLBfloat16Type(ET))
        ToExpandVIDWithSYCLTypeByValComp.push_back(&F);
      else
        llvm_unreachable("The component argument type of an "
                         "VectorInsertDynamic instruction can't be a "
                         "structure other than SYCL half.");
    }
  }

  for (auto *F : ToExpandVEDWithSYCLTypeSRetArg)
    expandVEDWithSYCLTypeSRetArg(F);
  for (auto *F : ToExpandVIDWithSYCLTypeByValComp)
    expandVIDWithSYCLTypeByValComp(F);
}

Value *SPIRVRegularizeLLVMBase::extendBitInstBoolArg(Instruction *II) {
  IRBuilder<> Builder(II);
  auto *ArgTy = II->getOperand(0)->getType();
  Type *NewArgType = nullptr;
  if (ArgTy->isIntegerTy()) {
    NewArgType = Builder.getInt32Ty();
  } else if (ArgTy->isVectorTy() &&
             cast<VectorType>(ArgTy)->getElementType()->isIntegerTy()) {
    unsigned NumElements = cast<FixedVectorType>(ArgTy)->getNumElements();
    NewArgType = VectorType::get(Builder.getInt32Ty(), NumElements, false);
  } else {
    llvm_unreachable("Unexpected type");
  }
  auto *NewBase = Builder.CreateZExt(II->getOperand(0), NewArgType);
  auto *NewShift = Builder.CreateZExt(II->getOperand(1), NewArgType);
  switch (II->getOpcode()) {
  case Instruction::LShr:
    return Builder.CreateLShr(NewBase, NewShift);
  case Instruction::Shl:
    return Builder.CreateShl(NewBase, NewShift);
  default:
    return II;
  }
}

bool SPIRVRegularizeLLVMBase::runRegularizeLLVM(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter SPIRVRegularizeLLVM:\n");
  regularize();
  LLVM_DEBUG(dbgs() << "After SPIRVRegularizeLLVM:\n" << *M);

  verifyRegularizationPass(*M, "SPIRVRegularizeLLVM");

  return true;
}

/// Remove entities not representable by SPIR-V
bool SPIRVRegularizeLLVMBase::regularize() {
  eraseUselessFunctions(M);
  expandSYCLTypeUsing(M);

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
          if (CF && CF->isIntrinsic()) {
            removeFnAttr(Call, Attribute::NoUnwind);
            auto *II = cast<IntrinsicInst>(Call);
            if (II->getIntrinsicID() == Intrinsic::memset ||
                II->getIntrinsicID() == Intrinsic::bswap)
              lowerIntrinsicToFunction(II);
            else if (II->getIntrinsicID() == Intrinsic::fshl ||
                     II->getIntrinsicID() == Intrinsic::fshr)
              lowerFunnelShift(II);
            else if (II->getIntrinsicID() == Intrinsic::umul_with_overflow)
              lowerUMulWithOverflow(II);
          }
        }

        // Translator treats i1 as boolean, but bit instructions take
        // a scalar/vector integers, so we have to extend such arguments
        if (II.isLogicalShift() &&
            II.getOperand(0)->getType()->isIntOrIntVectorTy(1)) {
          auto *NewInst = extendBitInstBoolArg(&II);
          for (auto *U : II.users()) {
            if (cast<Instruction>(U)->getOpcode() == Instruction::ZExt) {
              U->dropAllReferences();
              U->replaceAllUsesWith(NewInst);
              ToErase.push_back(cast<Instruction>(U));
            }
          }
          ToErase.push_back(&II);
        }

        // Remove optimization info not supported by SPIRV
        if (auto BO = dyn_cast<BinaryOperator>(&II)) {
          if (isa<PossiblyExactOperator>(BO) && BO->isExact())
            BO->setIsExact(false);
        }

        // FIXME: This is not valid handling for freeze instruction
        if (auto FI = dyn_cast<FreezeInst>(&II)) {
          FI->replaceAllUsesWith(FI->getOperand(0));
          FI->dropAllReferences();
          ToErase.push_back(FI);
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
        // Add an additional bitcast in case address space cast also changes
        // pointer element type.
        if (auto *ASCast = dyn_cast<AddrSpaceCastInst>(&II)) {
          Type *DestTy = ASCast->getDestTy();
          Type *SrcTy = ASCast->getSrcTy();
          if (!II.getContext().supportsTypedPointers())
            continue;
          if (DestTy->getScalarType()->getNonOpaquePointerElementType() !=
              SrcTy->getScalarType()->getNonOpaquePointerElementType()) {
            Type *InterTy = PointerType::getWithSamePointeeType(
                cast<PointerType>(DestTy->getScalarType()),
                cast<PointerType>(SrcTy->getScalarType())
                    ->getPointerAddressSpace());
            if (DestTy->isVectorTy())
              InterTy = VectorType::get(
                  InterTy, cast<VectorType>(DestTy)->getElementCount());
            BitCastInst *NewBCast = new BitCastInst(
                ASCast->getPointerOperand(), InterTy, /*NameStr=*/"", ASCast);
            AddrSpaceCastInst *NewASCast =
                new AddrSpaceCastInst(NewBCast, DestTy, /*NameStr=*/"", ASCast);
            ToErase.push_back(ASCast);
            ASCast->dropAllReferences();
            ASCast->replaceAllUsesWith(NewASCast);
          }
        }
        if (auto Cmpxchg = dyn_cast<AtomicCmpXchgInst>(&II)) {
          // Transform:
          // %1 = cmpxchg i32* %ptr, i32 %comparator, i32 %0 seq_cst acquire
          // To:
          // %cmpxchg.res = call spir_func
          //   i32 @_Z29__spirv_AtomicCompareExchangePiiiiii(
          //   i32* %ptr, i32 1, i32 16, i32 2, i32 %0, i32 %comparator)
          // %cmpxchg.success = icmp eq i32 %cmpxchg.res, %comparator
          // %1 = insertvalue { i32, i1 } undef, i32 %cmpxchg.res, 0
          // %2 = insertvalue { i32, i1 } %1, i1 %cmpxchg.success, 1

          // To get memory scope argument we might use Cmpxchg->getSyncScopeID()
          // but LLVM's cmpxchg instruction is not aware of OpenCL(or SPIR-V)
          // memory scope enumeration. And assuming the produced SPIR-V module
          // will be consumed in an OpenCL environment, we can use the same
          // memory scope as OpenCL atomic functions that do not have
          // memory_scope argument, i.e. memory_scope_device. See the OpenCL C
          // specification p6.13.11. Atomic Functions

          // cmpxchg LLVM instruction returns a pair {i32, i1}: the original
          // value and a flag indicating success (true) or failure (false).
          // OpAtomicCompareExchange SPIR-V instruction returns only the
          // original value. To keep the return type({i32, i1}) we construct
          // a composite. The first element of the composite holds result of
          // OpAtomicCompareExchange, i.e. the original value. The second
          // element holds result of comparison of the returned value and the
          // comparator, which matches with semantics of the flag returned by
          // cmpxchg.
          Value *Ptr = Cmpxchg->getPointerOperand();
          Value *MemoryScope = getInt32(M, spv::ScopeDevice);
          auto SuccessOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getSuccessOrdering()));
          auto FailureOrder = static_cast<OCLMemOrderKind>(
              llvm::toCABI(Cmpxchg->getFailureOrdering()));
          Value *EqualSem = getInt32(M, OCLMemOrderMap::map(SuccessOrder));
          Value *UnequalSem = getInt32(M, OCLMemOrderMap::map(FailureOrder));
          Value *Val = Cmpxchg->getNewValOperand();
          Value *Comparator = Cmpxchg->getCompareOperand();

          Type *MemType = Cmpxchg->getCompareOperand()->getType();

          llvm::Value *Args[] = {Ptr,        MemoryScope, EqualSem,
                                 UnequalSem, Val,         Comparator};
          auto *Res =
              addCallInstSPIRV(M, "__spirv_AtomicCompareExchange", MemType,
                               Args, nullptr, {MemType}, &II, "cmpxchg.res");
          IRBuilder<> Builder(Cmpxchg);
          auto *Cmp = Builder.CreateICmpEQ(Res, Comparator, "cmpxchg.success");
          auto *V1 = Builder.CreateInsertValue(
              UndefValue::get(Cmpxchg->getType()), Res, 0);
          auto *V2 = Builder.CreateInsertValue(V1, Cmp, 1, Cmpxchg->getName());
          Cmpxchg->replaceAllUsesWith(V2);
          ToErase.push_back(Cmpxchg);
        }
      }
    }
    for (Instruction *V : ToErase) {
      assert(V->user_empty() && "User non-empty\n");
      V->eraseFromParent();
    }
  }

  if (SPIRVDbgSaveRegularizedModule)
    saveLLVMModule(M, RegularizedModuleTmpFile);
  return true;
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVRegularizeLLVMLegacy, "spvregular",
                "Regularize LLVM for SPIR-V", false, false)

ModulePass *llvm::createSPIRVRegularizeLLVMLegacy() {
  return new SPIRVRegularizeLLVMLegacy();
}
