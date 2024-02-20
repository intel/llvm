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
    if (F.getName().starts_with("_Z28__spirv_VectorExtractDynamic") &&
        F.hasStructRetAttr()) {
      auto *SRetTy = F.getParamStructRetType(0);
      if (isSYCLHalfType(SRetTy) || isSYCLBfloat16Type(SRetTy))
        ToExpandVEDWithSYCLTypeSRetArg.push_back(&F);
      else
        llvm_unreachable("The return type of the VectorExtractDynamic "
                         "instruction cannot be a structure other than SYCL "
                         "half.");
    }
    if (F.getName().starts_with("_Z27__spirv_VectorInsertDynamic") &&
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

bool SPIRVRegularizeLLVMBase::runRegularizeLLVM(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter SPIRVRegularizeLLVM:\n");
  regularize();
  LLVM_DEBUG(dbgs() << "After SPIRVRegularizeLLVM:\n" << *M);

  verifyRegularizationPass(*M, "SPIRVRegularizeLLVM");

  return true;
}

// This is a temporary workaround to deal with a graphics driver failure not
// able to support the typed pointer reverse translation of
// getelementptr i8, ptr @__spirv_Builtin* patterns. This replaces such
// accesses with getelementptr i32, ptr @__spirv_Builtin instead.
static void simplifyBuiltinVarAccesses(GlobalValue *GV) {
  // IGC only supports:
  // load GV
  // load (addrspacecast GV)
  // load (gep (addrspacecast GV))
  // load (gep GV)
  // Opaque pointers will cause the optimizer to use i8 geps, or to remove
  // 0-index geps entirely (adding bitcasts to the result). Restore these to
  // avoid bitcasts in the resulting IR.
  if (GV->getContext().supportsTypedPointers())
    return;

  Type *Ty = GV->getValueType();
  Type *ScalarTy = Ty->getScalarType();
  SmallVector<Value *, 4> Users;
  for (auto User : GV->users()) {
    if (auto *LI = dyn_cast<LoadInst>(User)) {
      if (LI->getType() != Ty)
        Users.push_back(LI);
    } else if (auto *GEP = dyn_cast<GEPOperator>(User)) {
      if (GEP->getSourceElementType() != Ty)
        Users.push_back(GEP);
    }
  }

  Type *Int32Ty = Type::getInt32Ty(GV->getContext());
  auto GetGep = [&](unsigned Offset) {
    return ConstantExpr::getGetElementPtr(
        Ty, GV,
        ArrayRef<Constant *>(
            {ConstantInt::get(Int32Ty, 0), ConstantInt::get(Int32Ty, Offset)}),
        true, 0);
  };

  const DataLayout &DL = GV->getParent()->getDataLayout();
  for (auto *User : Users) {
    if (auto *LI = dyn_cast<LoadInst>(User)) {
      LI->setOperand(0, GetGep(0));
    } else if (auto *GEP = dyn_cast<GEPOperator>(User)) {
      APInt Offset(64, 0);
      GEP->accumulateConstantOffset(DL, Offset);
      APInt Index;
      uint64_t Remainder;
      APInt::udivrem(Offset, ScalarTy->getScalarSizeInBits() / 8, Index,
                     Remainder);
      assert(Remainder == 0 && "Cannot handle misaligned access to builtins");
      GEP->replaceAllUsesWith(GetGep(Index.getZExtValue()));
      if (auto *Inst = dyn_cast<Instruction>(GEP))
        Inst->eraseFromParent();
    }
  }
}

/// Remove entities not representable by SPIR-V
bool SPIRVRegularizeLLVMBase::regularize() {
  eraseUselessFunctions(M);
  expandSYCLTypeUsing(M);

  for (auto &GV : M->globals()) {
    SPIRVBuiltinVariableKind Kind;
    if (isSPIRVBuiltinVariable(&GV, &Kind))
      simplifyBuiltinVarAccesses(&GV);
  }

  // Kernels called by other kernels
  std::vector<Function *> CalledKernels;
  for (auto I = M->begin(), E = M->end(); I != E;) {
    Function *F = &(*I++);
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      continue;
    }

    std::vector<Instruction *> ToErase;
    for (BasicBlock &BB : *F) {
      for (Instruction &II : BB) {
        if (auto *Call = dyn_cast<CallInst>(&II)) {
          Call->setTailCall(false);
          Function *CF = Call->getCalledFunction();
          if (CF && CF->getCallingConv() == CallingConv::SPIR_KERNEL) {
            CalledKernels.push_back(CF);
          } else if (CF && CF->isIntrinsic()) {
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

        if (II.isLogicalShift()) {
          // Translator treats i1 as boolean, but bit instructions take
          // a scalar/vector integers, so we have to extend such arguments.
          // shl i1 %a %b and lshr i1 %a %b are now converted on:
          // %0 = select i1 %a, i32 1, i32 0
          // %1 = select i1 %b, i32 1, i32 0
          // %2 = lshr i32 %0, %1
          // if any other instruction other than zext was dependant:
          // %3 = icmp ne i32 %2, 0
          // which converts it back to i1 and replace original result with %3
          // to dependant instructions.
          if (II.getOperand(0)->getType()->isIntOrIntVectorTy(1)) {
            IRBuilder<> Builder(&II);
            Value *CmpNEInst = nullptr;
            Constant *ConstZero = ConstantInt::get(Builder.getInt32Ty(), 0);
            Constant *ConstOne = ConstantInt::get(Builder.getInt32Ty(), 1);
            if (auto *VecTy =
                    dyn_cast<FixedVectorType>(II.getOperand(0)->getType())) {
              const unsigned NumElements = VecTy->getNumElements();
              ConstZero = ConstantVector::getSplat(
                  ElementCount::getFixed(NumElements), ConstZero);
              ConstOne = ConstantVector::getSplat(
                  ElementCount::getFixed(NumElements), ConstOne);
            }
            Value *ExtendedBase =
                Builder.CreateSelect(II.getOperand(0), ConstOne, ConstZero);
            Value *ExtendedShift =
                Builder.CreateSelect(II.getOperand(1), ConstOne, ConstZero);
            Value *ExtendedShiftedVal =
                Builder.CreateLShr(ExtendedBase, ExtendedShift);
            SmallVector<User *, 8> Users(II.users());
            for (User *U : Users) {
              if (auto *UI = dyn_cast<Instruction>(U)) {
                if (UI->getOpcode() == Instruction::ZExt) {
                  UI->dropAllReferences();
                  UI->replaceAllUsesWith(ExtendedShiftedVal);
                  ToErase.push_back(UI);
                  continue;
                }
              }
              if (!CmpNEInst) {
                CmpNEInst = Builder.CreateICmpNE(ExtendedShiftedVal, ConstZero);
              }
              U->replaceUsesOfWith(&II, CmpNEInst);
            }
            ToErase.push_back(&II);
          }
        }

        // Remove optimization info not supported by SPIRV
        if (auto *BO = dyn_cast<BinaryOperator>(&II)) {
          if (isa<PossiblyExactOperator>(BO) && BO->isExact())
            BO->setIsExact(false);
        }

        // FIXME: This is not valid handling for freeze instruction
        if (auto *FI = dyn_cast<FreezeInst>(&II)) {
          auto *V = FI->getOperand(0);
          if (isa<UndefValue>(V))
            V = Constant::getNullValue(V->getType());
          FI->replaceAllUsesWith(V);
          FI->dropAllReferences();
          ToErase.push_back(FI);
        }

        // Remove metadata not supported by SPIRV
        static const char *MDs[] = {
            "tbaa",
            "range",
        };
        for (auto &MDName : MDs) {
          if (II.getMetadata(MDName)) {
            II.setMetadata(MDName, nullptr);
          }
        }
        if (auto *Cmpxchg = dyn_cast<AtomicCmpXchgInst>(&II)) {
          // Transform:
          // %1 = cmpxchg i32* %ptr, i32 %comparator, i32 %0 seq_cst acquire
          // To:
          // %cmpxchg.res = call spir_func
          //   i32 @_Z29__spirv_AtomicCompareExchangePiiiiii(
          //   i32* %ptr, i32 1, i32 16, i32 2, i32 %0, i32 %comparator)
          // %cmpxchg.success = icmp eq i32 %cmpxchg.res, %comparator
          // %1 = insertvalue { i32, i1 } undef, i32 %cmpxchg.res, 0
          // %2 = insertvalue { i32, i1 } %1, i1 %cmpxchg.success, 1

          // To get memory scope argument we use Cmpxchg->getSyncScopeID()
          // but LLVM's cmpxchg instruction is not aware of OpenCL(or SPIR-V)
          // memory scope enumeration. If the scope is not set and assuming the
          // produced SPIR-V module will be consumed in an OpenCL environment,
          // we can use the same memory scope as OpenCL atomic functions that do
          // not have memory_scope argument, i.e. memory_scope_device. See the
          // OpenCL C specification p6.13.11. Atomic Functions

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
          SmallVector<StringRef> SSIDs;
          Cmpxchg->getContext().getSyncScopeNames(SSIDs);

          spv::Scope S;
          // Fill unknown syncscope value to default Device scope.
          if (!OCLStrMemScopeMap::find(SSIDs[Cmpxchg->getSyncScopeID()].str(),
                                       &S)) {
            S = ScopeDevice;
          }
          Value *MemoryScope = getInt32(M, S);
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
