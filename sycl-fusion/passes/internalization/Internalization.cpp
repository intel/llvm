//==------------------------- Internalization.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Internalization.h"

#include <numeric>
#include <sstream>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "cleanup/Cleanup.h"
#include "debug/PassDebug.h"
#include "metadata/MDParsing.h"
#include "target/TargetFusionInfo.h"

#define DEBUG_TYPE "sycl-fusion"

using namespace llvm;
using namespace PatternMatch;

constexpr static StringLiteral PrivatePromotion{"private"};
constexpr static StringLiteral LocalPromotion{"local"};
constexpr static StringLiteral NoPromotion{"none"};

///
/// Helper struct to capture number of elements and their size in bytes.
struct PromotionInfo {
  std::size_t LocalSize;
  std::size_t ElemSize;
};

///
/// Helper function implementing private and public internalization.
///
/// Can be configured using the given parameters.
struct SYCLInternalizerImpl {
  /// Address space to use when promoting.
  unsigned AS;
  /// What value to handle.
  StringRef Kind;
  /// Whether or not to create allocas.
  bool CreateAllocas;
  /// Interface to target-specific information.
  TargetFusionInfo TargetInfo;

  /// Implements internalization the pass run.
  PreservedAnalyses operator()(Module &M, ModuleAnalysisManager &AM) const;

  ///
  /// Update a value to be promoted in a function.
  /// This runs after analysis, so all conditions for promotion should be
  /// fulfilled.
  ///
  /// Traverse all of the value's users:
  /// - The value appears in a GEP/bitcast instruction: mutate the instruction
  /// type and update it in the same way.
  /// - The value appears in a call:
  ///   - Promote the function and call the new function instead,
  ///   keeping the original function.
  /// - The value appears in a load/store operation: Do nothing
  void promoteValue(Value *Val, const PromotionInfo &PromInfo,
                    bool InAggregate) const;

  void promoteGEPI(GetElementPtrInst *GEPI, const Value *Val,
                   const PromotionInfo &PromInfo, bool InAggregate) const;

  void promoteCall(CallBase *C, const Value *Val,
                   const PromotionInfo &PromInfo) const;

  ///
  /// Function to promote a set of arguments from a function.
  /// This runs after analysis, so all conditions for promotion should be
  /// fulfilled.
  ///
  /// 1. Declare the new promoted function with the updated signature.
  /// 2. Clone the function with the desired promoted arguments.
  /// 3. If required, erase the old function.
  Function *promoteFunctionArgs(Function *F, ArrayRef<PromotionInfo> PromInfos,
                                bool CreateAllocas,
                                bool KeepOriginal = false) const;

  ///
  /// Check that an value can be promoted.
  /// For GEP and Call instructions, delegate to the specific implementations.
  /// \p InAggregate indicates that at least one GEP instruction addressing into
  /// an aggregate object was encountered, hence \p Val no longer represents a
  /// pure offset computation on the original candidate argument.
  /// For address-space casts, pointer-to-int conversions and unknown users,
  /// return an error.
  Error canPromoteValue(Value *Val, const PromotionInfo &PromInfo,
                        bool InAggregate) const;

  ///
  /// Check that the operand of a GEP can be promoted to its users, and
  /// propagate whether it represents a pointer into an aggregate object.
  Error canPromoteGEP(GetElementPtrInst *GEPI, const Value *Val,
                      const PromotionInfo &PromInfo, bool InAggregate) const;

  ///
  /// Check if operand to a function call can be promoted.
  /// If the function returns a pointer, or the operand points into an aggregate
  /// object, return an error. Otherwise, check if the corresponding formal
  /// parameter can be promoted in the function body.
  Error canPromoteCall(CallBase *C, const Value *Val,
                       const PromotionInfo &PromInfo, bool InAggregate) const;

  Error checkArgsPromotable(Function *F,
                            SmallVectorImpl<PromotionInfo> &PromInfos) const;
};

constexpr StringLiteral SYCLInternalizer::Key;
constexpr StringLiteral SYCLInternalizer::LocalSizeKey;
constexpr StringLiteral SYCLInternalizer::ElemSizeKey;

static Expected<SmallVector<PromotionInfo>>
getInternalizationFromMD(Function *F, StringRef Kind) {
  SmallVector<PromotionInfo> Info;
  MDNode *MD = F->getMetadata(SYCLInternalizer::Key);
  MDNode *LSMD = F->getMetadata(SYCLInternalizer::LocalSizeKey);
  MDNode *ESMD = F->getMetadata(SYCLInternalizer::ElemSizeKey);
  if (!MD || !LSMD || !ESMD) {
    return createStringError(inconvertibleErrorCode(),
                             "Promotion metadata not available");
  }
  for (auto I : zip(MD->operands(), LSMD->operands(), ESMD->operands())) {
    const auto *MDS = cast<MDString>(std::get<0>(I));
    const auto Val = [&]() -> PromotionInfo {
      if (MDS->getString() == Kind) {
        auto LS = metadataToUInt<std::size_t>(std::get<1>(I));
        if (auto Err = LS.takeError()) {
          // Do nothing
          handleAllErrors(std::move(Err), [](const StringError &SE) {
            FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
          });
          return {};
        }
        auto ES = metadataToUInt<std::size_t>(std::get<2>(I));
        if (auto Err = ES.takeError()) {
          // Do nothing
          handleAllErrors(std::move(Err), [](const StringError &SE) {
            FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
          });
          return {};
        }
        return {*LS, *ES};
      }
      return {};
    }();
    Info.emplace_back(Val);
  }
  return Info;
}

static void updateInternalizationMD(Function *F, StringRef Kind,
                                    ArrayRef<PromotionInfo> PromInfos) {
  MDNode *MD = F->getMetadata(SYCLInternalizer::Key);
  MDNode *LSMD = F->getMetadata(SYCLInternalizer::LocalSizeKey);
  MDNode *ESMD = F->getMetadata(SYCLInternalizer::ElemSizeKey);
  assert(MD && LSMD && "Promotion metadata not available");
  assert(MD->getNumOperands() == PromInfos.size() &&
         LSMD->getNumOperands() == PromInfos.size() &&
         ESMD->getNumOperands() == PromInfos.size() &&
         "Size mismatch in promotion metadata");
  for (auto I : enumerate(PromInfos)) {
    const auto *CurMDS = cast<MDString>(MD->getOperand(I.index()));
    if (CurMDS->getString() == Kind) {
      if (I.value().LocalSize == 0) {
        // The metadata indicates that this argument should be promoted, but the
        // analysis has deemed this infeasible (local size after analysis is 0).
        // Update the metadata-entry for this argument.
        auto *NewMDS = MDString::get(F->getContext(), NoPromotion);
        MD->replaceOperandWith(I.index(), NewMDS);
        auto *EmptyStr = MDString::get(F->getContext(), "");
        LSMD->replaceOperandWith(I.index(), EmptyStr);
        ESMD->replaceOperandWith(I.index(), EmptyStr);
      }
    }
  }
}

///
/// If \p GEPI represents a constant offset in bytes, return it, otherwise
/// return an empty value.
static std::optional<unsigned> getConstantByteOffset(GetElementPtrInst *GEPI,
                                                     const DataLayout &DL) {
  MapVector<Value *, APInt> VariableOffsets;
  auto IW = DL.getIndexSizeInBits(GEPI->getPointerAddressSpace());
  APInt ConstantOffset = APInt::getZero(IW);
  if (GEPI->collectOffset(DL, IW, VariableOffsets, ConstantOffset) &&
      VariableOffsets.empty()) {
    return ConstantOffset.getZExtValue();
  }
  return {};
}

///
/// When performing internalization, GEP instructions must be remapped, as the
/// address space has changed from N to N / LocalSize.
static void remap(GetElementPtrInst *GEPI, const PromotionInfo &PromInfo) {
  IRBuilder<> Builder{GEPI};

  if (PromInfo.LocalSize == 1) {
    // Squash the index and let instcombine clean-up afterwards.
    GEPI->idx_begin()->set(Builder.getInt64(0));
    return;
  }

  // GEPs with constant offset may be marked for remapping even if their element
  // size differs from the accessor's element size. However we know that the
  // offset is a multiple of the latter. Rewrite the instruction to represent a
  // number of _elements_ to make it compatible with other GEPs in the current
  // chain.
  auto &DL = GEPI->getModule()->getDataLayout();
  auto SrcElemTySz = DL.getTypeAllocSize(GEPI->getSourceElementType());
  if (SrcElemTySz != PromInfo.ElemSize) {
    auto COff = getConstantByteOffset(GEPI, DL);
    // This is special case #2 in `getGEPKind`.
    assert(COff.has_value() && *COff % PromInfo.ElemSize == 0 &&
           GEPI->getNumIndices() == 1);
    auto *IntTypeWithSameWidthAsAccessorElementType =
        Builder.getIntNTy(PromInfo.ElemSize * 8);
    GEPI->setSourceElementType(IntTypeWithSameWidthAsAccessorElementType);
    GEPI->setResultElementType(IntTypeWithSameWidthAsAccessorElementType);
    GEPI->idx_begin()->set(Builder.getInt64(*COff / PromInfo.ElemSize));
  }

  // An individual `GEP(ptr, offset)` is rewritten as
  // `GEP(ptr, offset % LocalSize)`.
  //
  // However, we often encounter chains of single-index GEPs:
  // ```
  // a = GEP ptr, off_1
  // b = GEP a, off_2
  // c = GEP b, off_3
  // ```
  //
  // These must be rewritten as:
  // ```
  // a = GEP ptr, off_1 % LocalSize
  // b = GEP ptr, (off_1 + off_2) % LocalSize
  // c = GEP ptr, (off_1 + off_2 + off_3) % LocalSize
  // ```
  //
  // This method is called during a def-use-traversal, i.e. `GEPI`'s pointer
  // operand has already been visited. Modular arithmetic satisfies the
  // following equation:
  // ```
  // ((x mod n) + y) mod n = (x + y) mod n
  // ```
  // Together, this means we can propagate the predecessor GEP's pointer
  // operand, and take its already wrapped offset, add `GEPI`'s offset, and wrap
  // the result again around LocalSize.
  //
  // If the predecessor is not a GEP, then we just rewrap `GEPI`'s index.
  Value *Dividend =
      TypeSwitch<Value *, Value *>(GEPI->getPointerOperand())
          .Case<GetElementPtrInst>([&](auto Pred) {
            GEPI->op_begin()->set(Pred->getPointerOperand());
            return Builder.CreateAdd(*Pred->idx_begin(), *GEPI->idx_begin());
          })
          .Default(GEPI->idx_begin()->get());
  Value *Remainder =
      Builder.CreateURem(Dividend, Builder.getInt64(PromInfo.LocalSize));
  GEPI->idx_begin()->set(Remainder);
}

///
/// Function to get the indices of a user in which a value appears.
static SmallVector<PromotionInfo>
getUsagesInternalization(const User *U, const Value *V,
                         const PromotionInfo &PromInfo) {
  SmallVector<PromotionInfo> InternInfo;
  std::transform(
      U->op_begin(), U->op_end(), std::back_inserter(InternInfo),
      [&](const Use &Us) { return Us == V ? PromInfo : PromotionInfo{}; });
  return InternInfo;
}

Error SYCLInternalizerImpl::canPromoteCall(CallBase *C, const Value *Val,
                                           const PromotionInfo &PromInfo,
                                           bool InAggregate) const {
  if (isa<PointerType>(C->getType())) {
    // With opaque pointers, we do not have the necessary information to compare
    // the element-type of the pointer returned by the function and the element
    // type of the pointer to promote. Therefore, we assume that the function
    // cannot be promoted if the function returns a pointer.
    return createStringError(
        inconvertibleErrorCode(),
        "It is not safe to promote a called function which returns a pointer.");
  }
  if (InAggregate) {
    return createStringError(
        inconvertibleErrorCode(),
        "Promotion of a pointer into an aggregate object to a called function "
        "is currently not supported.");
  }

  SmallVector<PromotionInfo> InternInfo =
      getUsagesInternalization(C, Val, PromInfo);
  assert(!InternInfo.empty() && "Value must be used at least once");
  if (auto Err = checkArgsPromotable(C->getCalledFunction(), InternInfo)) {
    return Err;
  }
  return Error::success();
}

enum GEPKind { INVALID = 0, NEEDS_REMAPPING, ADDRESSES_INTO_AGGREGATE };

static int getGEPKind(GetElementPtrInst *GEPI, const PromotionInfo &PromInfo) {
  assert(GEPI->getNumIndices() >= 1 && "No-op GEP encountered");

  // Inspect the GEP's source element type.
  auto &DL = GEPI->getModule()->getDataLayout();
  auto SrcElemTySz = DL.getTypeAllocSize(GEPI->getSourceElementType());

  // `GEPI`'s first index is selecting elements. Unless it is constant zero, we
  // have to remap. If there are more indices, we start to address into an
  // aggregate type.
  if (SrcElemTySz == PromInfo.ElemSize) {
    int Kind = INVALID;
    if (!match(GEPI->idx_begin()->get(), m_ZeroInt()))
      Kind |= NEEDS_REMAPPING;
    if (GEPI->getNumIndices() >= 2)
      Kind |= ADDRESSES_INTO_AGGREGATE;
    assert(Kind != INVALID && "No-op GEP encountered");
    return Kind;
  }

  // We can handle a mismatch between `GEPI`'s element size and the accessors
  // element size if `GEPI` represents a constant offset.
  if (auto COff = getConstantByteOffset(GEPI, DL)) {
    if (*COff < PromInfo.ElemSize) {
      // Special case #1: The offset is less than the element size, hence we're
      // addressing into an aggregrate and no remapping is required.
      return ADDRESSES_INTO_AGGREGATE;
    }
    if (*COff % PromInfo.ElemSize == 0 && GEPI->getNumIndices() == 1) {
      // Special case #2: The offset is a multiple of the element size, meaning
      // `GEPI` selects an element and is subject to remapping.
      return NEEDS_REMAPPING;
    }
  }

  // We don't know what `GEPI` addresses; bail out.
  return INVALID;
}

Error SYCLInternalizerImpl::canPromoteGEP(GetElementPtrInst *GEPI,
                                          const Value *Val,
                                          const PromotionInfo &PromInfo,
                                          bool InAggregate) const {
  if (cast<PointerType>(GEPI->getType())->getAddressSpace() == AS) {
    // If the GEPI is already using the correct address-space, no change is
    // required.
    return Error::success();
  }

  // Inspect the current instruction.
  auto Kind = getGEPKind(GEPI, PromInfo);
  if (Kind == INVALID) {
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported pointer arithmetic");
  }

  // Recurse to check all users of the GEP.
  return canPromoteValue(GEPI, PromInfo,
                         InAggregate || (Kind & ADDRESSES_INTO_AGGREGATE));
}

Error SYCLInternalizerImpl::canPromoteValue(Value *Val,
                                            const PromotionInfo &PromInfo,
                                            bool InAggregate) const {
  for (auto *U : Val->users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot promote value used in a place other than an instruction");
    }
    switch (I->getOpcode()) {
    case Instruction::AddrSpaceCast:
      return createStringError(inconvertibleErrorCode(),
                               "It is not safe to promote values appearing "
                               "in an addrspacecast operation");
    case Instruction::PtrToInt:
      return createStringError(inconvertibleErrorCode(),
                               "It is not safe to promote values appearing "
                               "in a ptrtoint operation");
    case Instruction::Call:
    case Instruction::Invoke:
    case Instruction::CallBr:
      if (auto Err =
              canPromoteCall(cast<CallBase>(I), Val, PromInfo, InAggregate)) {
        return Err;
      }
      break;
    case Instruction::GetElementPtr:
      if (auto Err = canPromoteGEP(cast<GetElementPtrInst>(I), Val, PromInfo,
                                   InAggregate)) {
        return Err;
      }
      break;
    case Instruction::Load:
      // Do not need to change anything here.
      break;
    case Instruction::Store:
      if (Val == cast<StoreInst>(I)->getValueOperand()) {
        return createStringError(
            inconvertibleErrorCode(),
            "It is not safe to promote values being stored to another pointer");
      }
      break;
    default:
      return createStringError(inconvertibleErrorCode(),
                               "Do not know how to handle value to promote");
    }
  }
  return Error::success();
}

Error SYCLInternalizerImpl::checkArgsPromotable(
    Function *F, SmallVectorImpl<PromotionInfo> &PromInfos) const {
  Error DeferredErrs = Error::success();
  for (auto I : enumerate(PromInfos)) {
    const auto &PromInfo = I.value();
    if (PromInfo.LocalSize == 0) {
      continue;
    }
    const size_t Index = I.index();
    Argument *Arg = F->getArg(Index);
    if (!isa<PointerType>(Arg->getType()) ||
        cast<Argument>(Arg)->hasByValAttr()) {
      // Omit non-pointer and byval arguments.
      PromInfos[Index].LocalSize = 0;
      continue;
    }
    if (auto Err = canPromoteValue(Arg, PromInfo, /*InAggregate=*/false)) {
      // Set the local size to 0 to indicate that this argument should not be
      // promoted.
      PromInfos[Index].LocalSize = 0;
      std::stringstream ErrorMessage;
      handleAllErrors(std::move(Err), [&](const StringError &SE) {
        ErrorMessage << "Failed to promote argument " << Index
                     << " of function " << F->getName().str() << ": "
                     << SE.getMessage() << "\n";
      });
      Error NewErr =
          createStringError(inconvertibleErrorCode(), ErrorMessage.str());
      DeferredErrs = joinErrors(std::move(DeferredErrs), std::move(NewErr));
    }
  }
  return DeferredErrs;
}

///
/// Function to perform the required cleaning actions.
static void cleanup(Function *OldF, Function *NewF, bool KeepOriginal,
                    const TargetFusionInfo &TFI) {
  if (!KeepOriginal) {
    NewF->takeName(OldF);
    TFI.notifyFunctionsDelete(OldF);
    OldF->eraseFromParent();
  }
  TFI.addKernelFunction(NewF);
}

void SYCLInternalizerImpl::promoteCall(CallBase *C, const Value *Val,
                                       const PromotionInfo &PromInfo) const {

  const SmallVector<PromotionInfo> InternInfo =
      getUsagesInternalization(C, Val, PromInfo);
  assert(!InternInfo.empty() && "Value must be used at least once");
  Function *NewF = promoteFunctionArgs(C->getCalledFunction(), InternInfo,
                                       /* CreateAllocas */ false,
                                       /*KeepOriginal*/ true);

  C->setCalledFunction(NewF);
}

void SYCLInternalizerImpl::promoteGEPI(GetElementPtrInst *GEPI,
                                       const Value *Val,
                                       const PromotionInfo &PromInfo,
                                       bool InAggregate) const {
  // Not PointerType is unreachable. Other case is caught in caller.
  if (cast<PointerType>(GEPI->getType())->getAddressSpace() != AS) {
    auto Kind = getGEPKind(GEPI, PromInfo);
    assert(Kind != INVALID);

    if (!InAggregate && (Kind & NEEDS_REMAPPING)) {
      remap(GEPI, PromInfo);
    }
    GEPI->mutateType(PointerType::get(GEPI->getContext(), AS));

    // Recurse to promote to all users of the GEP.
    return promoteValue(GEPI, PromInfo,
                        InAggregate || (Kind & ADDRESSES_INTO_AGGREGATE));
  }
}

void SYCLInternalizerImpl::promoteValue(Value *Val,
                                        const PromotionInfo &PromInfo,
                                        bool InAggregate) const {
  // Freeze the current list of users, as promoteGEPI re-links the elements in a
  // GEP chain, and hence may introduce new users to `Val`.
  SmallVector<User *> CurrentUsers{Val->users()};
  for (auto *U : CurrentUsers) {
    auto *I = cast<Instruction>(U);
    switch (I->getOpcode()) {
    case Instruction::Call:
    case Instruction::Invoke:
    case Instruction::CallBr:
      assert(!InAggregate);
      promoteCall(cast<CallBase>(I), Val, PromInfo);
      break;
    case Instruction::GetElementPtr:
      promoteGEPI(cast<GetElementPtrInst>(I), Val, PromInfo, InAggregate);
      break;
    case Instruction::Load:
    case Instruction::Store:
      // Do not need to change anything here.
      break;
    default:
      llvm_unreachable("Unknown user of value to promote");
    }
  }
}

///
/// Get promoted function type.
static FunctionType *getPromotedFunctionType(FunctionType *OrigTypes,
                                             ArrayRef<PromotionInfo> PromInfos,
                                             unsigned AS) {
  SmallVector<Type *> Types{OrigTypes->param_begin(), OrigTypes->param_end()};
  for (auto Arg : enumerate(PromInfos)) {
    // No internalization.
    if (Arg.value().LocalSize == 0) {
      continue;
    }
    Type *&Ty = Types[Arg.index()];
    // TODO: Catch this case earlier
    if (isa<PointerType>(Ty)) {
      Ty = PointerType::get(Ty->getContext(), AS);
    }
  }
  return FunctionType::get(OrigTypes->getReturnType(), Types,
                           OrigTypes->isVarArg());
}

static Function *
getPromotedFunctionDeclaration(Function *F, ArrayRef<PromotionInfo> PromInfos,
                               unsigned AS, bool ChangeTypes) {
  FunctionType *Ty = F->getFunctionType();
  // If we do not need to change the types, we just copy the function
  // declaration.
  FunctionType *NewTy =
      ChangeTypes ? getPromotedFunctionType(Ty, PromInfos, AS) : Ty;
  return Function::Create(NewTy, F->getLinkage(), F->getAddressSpace(),
                          F->getName(), F->getParent());
}

///
/// For private promotion, we want to replace each argument by an alloca.
Value *replaceByNewAlloca(Argument *Arg, unsigned AS,
                          const PromotionInfo &PromInfo) {
  IRBuilder<> Builder{
      &*Arg->getParent()->getEntryBlock().getFirstInsertionPt()};
  auto ArgAS = cast<PointerType>(Arg->getType())->getAddressSpace();
  auto *Alloca = Builder.CreateAlloca(
      Builder.getInt8Ty(), ArgAS,
      Builder.getInt64(PromInfo.LocalSize * PromInfo.ElemSize));
  Alloca->setAlignment(Arg->getParamAlign().valueOrOne());
  Arg->replaceAllUsesWith(Alloca);
  Alloca->mutateType(Builder.getPtrTy(AS));
  return Alloca;
}

Function *SYCLInternalizerImpl::promoteFunctionArgs(
    Function *OldF, ArrayRef<PromotionInfo> PromInfos, bool CreateAllocas,
    bool KeepOriginal) const {
  // We first declare the promoted function with the new signature.
  Function *NewF =
      getPromotedFunctionDeclaration(OldF, PromInfos, AS,
                                     /*ChangeTypes*/ !CreateAllocas);

  // Clone the old function into the new function.
  {
    ValueMap<const Value *, WeakTrackingVH> VMap;
    for (auto SrcI = OldF->arg_begin(), DestI = NewF->arg_begin();
         SrcI != OldF->arg_end(); ++SrcI, ++DestI) {
      DestI->setName(SrcI->getName()); // Copy argument names.
      VMap[&*SrcI] = &*DestI;          // Map homologous arguments.
    }
    // We can omit the vector of returns.
    SmallVector<ReturnInst *> Returns;
    CloneFunctionInto(NewF, OldF, VMap, CloneFunctionChangeType::GlobalChanges,
                      Returns);
  }

  // Update each of the values to promote.
  SmallVector<bool> ArgIsPromoted(PromInfos.size());
  for (auto I : enumerate(PromInfos)) {
    const auto &PromInfo = I.value();
    if (PromInfo.LocalSize == 0) {
      continue;
    }
    const auto Index = I.index();
    Value *Arg{NewF->getArg(Index)};
    ArgIsPromoted[Index] = true;
    if (!isa<PointerType>(Arg->getType()) ||
        cast<Argument>(Arg)->hasByValAttr()) {
      // Omit non-pointer and byval arguments.
      continue;
    }
    if (CreateAllocas) {
      Arg = replaceByNewAlloca(cast<Argument>(Arg), AS, PromInfo);
    }
    promoteValue(Arg, PromInfo, /*InAggregate=*/false);
  }

  TargetInfo.updateAddressSpaceMetadata(NewF, ArgIsPromoted, AS);

  cleanup(OldF, NewF, KeepOriginal, TargetInfo);

  return NewF;
}

PreservedAnalyses
SYCLInternalizerImpl::operator()(Module &M, ModuleAnalysisManager &AM) const {
  bool Changed{false};
  SmallVector<Function *> ToUpdate;
  for (auto &F : M) {
    if (F.hasMetadata(SYCLInternalizer::Key)) {
      ToUpdate.emplace_back(&F);
    }
  }
  for (auto *F : ToUpdate) {
    Expected<SmallVector<PromotionInfo>> PromInfosOrErr =
        getInternalizationFromMD(F, Kind);
    if (auto E = PromInfosOrErr.takeError()) {
      handleAllErrors(std::move(E), [](const StringError &SE) {
        FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
      });
      continue;
    }
    SmallVector<PromotionInfo> &PromInfos = *PromInfosOrErr;

    // Analysis phase. Check which arguments, for which promotion was requested
    // by the user/runtime, can actually be promoted. The analysis will not
    // perform any modifications to the code. Instead, it sets the local size of
    // arguments, for which promotion would fail, to 0, so no promotion is
    // performned for this argument in the transformation phase.
    if (auto Err = checkArgsPromotable(F, PromInfos)) {
      FUSION_DEBUG(llvm::dbgs()
                   << "Unable to perform all promotions for function "
                   << F->getName() << ". Detailed information: \n");
      handleAllErrors(std::move(Err), [&](const StringError &SE) {
        FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
      });
    }

    // Update the internalization metadata after the analysis phase to remove
    // the internalization MD for arguments for which the analysis has deemed
    // the promotion infeasible.
    updateInternalizationMD(F, Kind, PromInfos);

    if (llvm::none_of(PromInfos, [](const auto &PromInfo) {
          return PromInfo.LocalSize > 0;
        })) {
      // If no arguments is requested & eligible for promotion after analysis,
      // skip the transformation phase.
      continue;
    }

    // Transformation phase. Promote the arguments for which it is actually safe
    // to perform promotion.
    promoteFunctionArgs(F, PromInfos, CreateAllocas);
    Changed = true;
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

static void moduleCleanup(Module &M, ModuleAnalysisManager &AM,
                          TargetFusionInfo &TFI) {
  SmallVector<Function *> ToProcess;
  for (auto &F : M) {
    if (F.hasMetadata(SYCLInternalizer::Key)) {
      ToProcess.emplace_back(&F);
    }
  }
  for (auto *F : ToProcess) {
    auto *MD = F->getMetadata(SYCLInternalizer::Key);
    // Use the argument usage mask to provide feedback to the runtime which
    // arguments have been promoted to private or local memory and which have
    // been eliminated in the process (private promotion).
    SmallVector<jit_compiler::ArgUsageUT> NewArgInfo;
    for (auto I : enumerate(MD->operands())) {
      const auto &MDS = cast<MDString>(I.value().get())->getString();
      if (MDS == PrivatePromotion) {
        NewArgInfo.push_back((jit_compiler::ArgUsage::PromotedPrivate |
                              jit_compiler::ArgUsage::Unused));
      } else if (MDS == LocalPromotion) {
        NewArgInfo.push_back((jit_compiler::ArgUsage::PromotedLocal |
                              jit_compiler::ArgUsage::Used));
      } else {
        NewArgInfo.push_back(jit_compiler::ArgUsage::Used);
      }
    }
    fullCleanup(NewArgInfo, F, AM, TFI,
                {SYCLInternalizer::Key, SYCLInternalizer::LocalSizeKey,
                 SYCLInternalizer::ElemSizeKey});
  }
}

PreservedAnalyses llvm::SYCLInternalizer::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  TargetFusionInfo TFI{&M};
  // Private promotion
  const PreservedAnalyses Tmp = SYCLInternalizerImpl{
      TFI.getPrivateAddressSpace(), PrivatePromotion, true, TFI}(M, AM);
  // Local promotion
  PreservedAnalyses Res = SYCLInternalizerImpl{
      TFI.getLocalAddressSpace(), LocalPromotion, false, TFI}(M, AM);

  Res.intersect(Tmp);

  if (!Res.areAllPreserved()) {
    moduleCleanup(M, AM, TFI);
  }
  return Res;
}
