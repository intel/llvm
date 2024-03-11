//==------------------------------ SYCLCP.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLCP.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/WithColor.h"

#include "cleanup/Cleanup.h"
#include "debug/PassDebug.h"
#include "metadata/MDParsing.h"

#define DEBUG_TYPE "sycl-fusion"

using namespace llvm;

constexpr StringLiteral SYCLCP::Key;

///
/// Codifies constants to be propagated as the index of the argument to be
/// replaced and a 64-bit value to be used instead.
struct ConstantInfo {
  unsigned Index;
  SmallVector<unsigned char> Value;

  ConstantInfo() = default;
  ConstantInfo(unsigned I, ArrayRef<unsigned char> Val)
      : Index{I}, Value{Val.begin(), Val.end()} {}
};

///
/// Reads constants from metadata.
static Expected<SmallVector<ConstantInfo>> getCPFromMD(Function *F) {
  SmallVector<ConstantInfo> Info;
  MDNode *MD = F->getMetadata(SYCLCP::Key);
  if (!MD) {
    return createStringError(inconvertibleErrorCode(),
                             "Constant progagation metadata not available");
  }
  for (auto I : enumerate(MD->operands())) {
    Expected<SmallVector<unsigned char>> Val =
        decodeConstantMetadata(I.value());
    if (auto Err = Val.takeError()) {
      // Do nothing
      handleAllErrors(std::move(Err), [](const StringError &SE) {
        FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
      });
      continue;
    }
    if (!Val->empty()) {
      Info.emplace_back(I.index(), Val.get());
    }
  }
  return Info;
}

///
/// Returns a constant of the given scalar type and value.
static Expected<Constant *> getConstantValue(const unsigned char **ValPtr,
                                             Type *Ty, bool ByVal) {
  if (Ty->isIntegerTy()) {
    unsigned NumBytes = Ty->getIntegerBitWidth() / 8;
    uint64_t IntValue = 0;
    // Copy only as many bytes as the type is actually wide.
    std::memcpy(&IntValue, *ValPtr, NumBytes);
    // Advance the pointer
    *ValPtr = *ValPtr + NumBytes;
    return ConstantInt::get(Ty, IntValue);
  }
  if (Ty->isDoubleTy()) {
    double DoubleValue = *(reinterpret_cast<const double *>(*ValPtr));
    // Advance the pointer
    *ValPtr = *ValPtr + sizeof(double);
    return ConstantFP::get(Ty, DoubleValue);
  }
  if (Ty->isFloatTy()) {
    double DoubleValue = *(reinterpret_cast<const float *>(*ValPtr));
    // Advance the pointer
    *ValPtr = *ValPtr + sizeof(float);
    return ConstantFP::get(Ty, DoubleValue);
  }
  return createStringError(inconvertibleErrorCode(),
                           "Only scalar and byval aggregate constants can be "
                           "propagated by -sycl-cp");
}

///
/// Initialize the members of an (potentially nested) aggregate constant through
/// store instructions.
static Error initializeAggregateConstant(const unsigned char **ValPtr,
                                         Type *CurrentTy, IRBuilder<> &Builder,
                                         Value *Alloca, Type *RootTy,
                                         ArrayRef<Value *> Indices) {
  if (CurrentTy->isIntegerTy() || CurrentTy->isFloatTy() ||
      CurrentTy->isDoubleTy()) {
    Expected<Value *> CVal = getConstantValue(ValPtr, CurrentTy, false);
    if (auto E = CVal.takeError()) {
      return E;
    }
    auto *GEP = Builder.CreateInBoundsGEP(RootTy, Alloca, Indices);
    Builder.CreateStore(CVal.get(), GEP);
    return Error::success();
  }
  if (CurrentTy->isArrayTy()) {
    ArrayType *ArrTy = cast<ArrayType>(CurrentTy);
    for (uint32_t I = 0; I < ArrTy->getArrayNumElements(); ++I) {
      SmallVector<Value *> ArrayIndices;
      ArrayIndices.insert(ArrayIndices.begin(), Indices.begin(), Indices.end());
      ArrayIndices.push_back(Builder.getInt32(I));
      Error RetCode =
          initializeAggregateConstant(ValPtr, ArrTy->getElementType(), Builder,
                                      Alloca, RootTy, ArrayIndices);
      if (RetCode) {
        return RetCode;
      }
    }
    return Error::success();
  }
  if (CurrentTy->isStructTy()) {
    StructType *StructTy = cast<StructType>(CurrentTy);
    SmallVector<Constant *> StructValues;
    for (auto IdxElem : llvm::enumerate(StructTy->elements())) {
      SmallVector<Value *> StructIndices;
      StructIndices.insert(StructIndices.begin(), Indices.begin(),
                           Indices.end());
      StructIndices.push_back(Builder.getInt32(IdxElem.index()));
      Error RetCode = initializeAggregateConstant(
          ValPtr, IdxElem.value(), Builder, Alloca, RootTy, StructIndices);
      if (RetCode) {
        return RetCode;
      }
    }
    return Error::success();
  }
  return createStringError(inconvertibleErrorCode(),
                           "Cannot construct constant aggregate");
}

///
/// Try to create an allocation and initialization via store instructions for a
/// aggregate constant argument passed byval.
static Expected<Value *> createAggregateConstant(const unsigned char **ValPtr,
                                                 Type *Ty,
                                                 IRBuilder<> &Builder) {
  auto *Alloca = Builder.CreateAlloca(Ty);
  Error RetCode = initializeAggregateConstant(ValPtr, Ty, Builder, Alloca, Ty,
                                              Builder.getInt32(0));
  if (RetCode) {
    return std::move(RetCode);
  }
  return Alloca;
}

///
/// Replaces evey use of each given argument by a constant.
static bool propagateConstants(Function *F, ArrayRef<ConstantInfo> Constants) {
  bool Changed{false};
  // Create aggregates constant allocations and initialization at the begin of
  // the function.
  IRBuilder<> Builder{&F->getEntryBlock().front()};
  MDNode *MD = F->getMetadata(SYCLCP::Key);
  auto *EmptyMDString = MDString::get(F->getContext(), "");
  for (const auto &C : Constants) {
    Argument *Arg = F->getArg(C.Index);
    const unsigned char *ValPtr = C.Value.data();
    Type *ArgTy = Arg->getType();
    Value *CVal;
    if (ArgTy->isPointerTy() && Arg->hasByValAttr()) {
      Expected<Value *> AggVal =
          createAggregateConstant(&ValPtr, Arg->getParamByValType(), Builder);
      if (auto E = AggVal.takeError()) {
        handleAllErrors(std::move(E), [](const StringError &SE) {
          FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
        });
        // Replace the MD operand with an empty string to signal the cleanup
        // that this argument has not been promoted.
        MD->replaceOperandWith(C.Index, EmptyMDString);
        continue;
      }
      CVal = AggVal.get();
    } else {
      Expected<Constant *> ScalarVal =
          getConstantValue(&ValPtr, ArgTy, Arg->hasByValAttr());
      if (auto E = ScalarVal.takeError()) {
        handleAllErrors(std::move(E), [](const StringError &SE) {
          FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
        });
        // Replace the MD operand with an empty string to signal the cleanup
        // that this argument has not been promoted.
        MD->replaceOperandWith(C.Index, EmptyMDString);
        continue;
      }
      CVal = ScalarVal.get();
    }
    // We simply
    Arg->replaceAllUsesWith(CVal);
    Changed = true;
  }
  return Changed;
}

static void moduleCleanup(Module &M, ModuleAnalysisManager &AM,
                          TargetFusionInfo &TFI) {
  SmallVector<Function *> ToProcess;
  for (auto &F : M) {
    if (F.hasMetadata(SYCLCP::Key)) {
      ToProcess.emplace_back(&F);
    }
  }
  for (auto *F : ToProcess) {
    auto *MD = F->getMetadata(SYCLCP::Key);
    SmallVector<jit_compiler::ArgUsageUT> NewArgInfo;
    for (auto I : enumerate(MD->operands())) {
      if (const auto *MDS = dyn_cast<MDString>(I.value().get())) {
        // A value is masked-out if it has a non-empty MDString
        if (MDS->getLength() > 0) {
          NewArgInfo.push_back(jit_compiler::ArgUsage::Unused);
          continue;
        }
      }
      NewArgInfo.push_back(jit_compiler::ArgUsage::Used);
    }
    fullCleanup(NewArgInfo, F, AM, TFI, {SYCLCP::Key});
  }
}

PreservedAnalyses SYCLCP::run(Module &M, ModuleAnalysisManager &AM) {
  bool Changed{false};
  SmallVector<Function *> ToUpdate;
  for (Function &F : M) {
    if (F.hasMetadata(Key)) {
      ToUpdate.emplace_back(&F);
    }
  }
  for (Function *F : ToUpdate) {
    Expected<SmallVector<ConstantInfo>> ConstantsOrErr = getCPFromMD(F);
    if (auto E = ConstantsOrErr.takeError()) {
      handleAllErrors(std::move(E), [](const StringError &SE) {
        FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
      });
      continue;
    }
    Changed = propagateConstants(F, *ConstantsOrErr) || Changed;
  }

  TargetFusionInfo TFI{&M};

  if (Changed) {
    moduleCleanup(M, AM, TFI);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
