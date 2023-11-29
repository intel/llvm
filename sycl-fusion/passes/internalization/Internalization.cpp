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
#include <llvm/Support/WithColor.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "cleanup/Cleanup.h"
#include "debug/PassDebug.h"
#include "metadata/MDParsing.h"
#include "target/TargetFusionInfo.h"

#define DEBUG_TYPE "sycl-fusion"

using namespace llvm;

constexpr static StringLiteral PrivatePromotion{"private"};
constexpr static StringLiteral LocalPromotion{"local"};
constexpr static StringLiteral NoPromotion{"none"};

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
  void promoteValue(Value *Val, std::size_t LocalSize, bool InAggregate) const;

  void promoteGEPI(GetElementPtrInst *GEPI, const Value *Val,
                   std::size_t LocalSize, bool InAggregate) const;

  void promoteCall(CallBase *C, const Value *Val, std::size_t LocalSize) const;

  ///
  /// Function to promote a set of arguments from a function.
  /// This runs after analysis, so all conditions for promotion should be
  /// fulfilled.
  ///
  /// 1. Declare the new promoted function with the updated signature.
  /// 2. Clone the function with the desired promoted arguments.
  /// 3. If required, erase the old function.
  Function *promoteFunctionArgs(Function *F,
                                ArrayRef<std::size_t> PromoteToLocal,
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
  Error canPromoteValue(Value *Val, size_t LocalSize, bool InAggregate) const;

  ///
  /// Check that the operand of a GEP can be promoted to its users, and
  /// propagate whether it represents a pointer into an aggregate object.
  Error canPromoteGEP(GetElementPtrInst *GEPI, const Value *Val,
                      size_t LocalSize, bool InAggregate) const;

  ///
  /// Check if operand to a function call can be promoted.
  /// If the function returns a pointer, or the operand points into an aggregate
  /// object, return an error. Otherwise, check if the corresponding formal
  /// parameter can be promoted in the function body.
  Error canPromoteCall(CallBase *C, const Value *Val, size_t LocalSize,
                       bool InAggregate) const;

  Error checkArgsPromotable(Function *F,
                            SmallVectorImpl<size_t> &PromoteArgSizes) const;
};

constexpr StringLiteral SYCLInternalizer::Key;
constexpr StringLiteral SYCLInternalizer::LocalSizeKey;

static Expected<SmallVector<std::size_t>>
getInternalizationFromMD(Function *F, StringRef Kind) {
  SmallVector<std::size_t> Info;
  MDNode *MD = F->getMetadata(SYCLInternalizer::Key);
  MDNode *LSMD = F->getMetadata(SYCLInternalizer::LocalSizeKey);
  if (!MD || !LSMD) {
    return createStringError(inconvertibleErrorCode(),
                             "Promotion metadata not available");
  }
  for (auto I : zip(MD->operands(), LSMD->operands())) {
    const auto *MDS = cast<MDString>(std::get<0>(I));
    const auto Val = [&]() -> std::size_t {
      if (MDS->getString() == Kind) {
        auto LS = metadataToUInt<std::size_t>(std::get<1>(I));
        if (auto Err = LS.takeError()) {
          // Do nothing
          handleAllErrors(std::move(Err), [](const StringError &SE) {
            FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
          });
          return 0;
        }
        return *LS;
      }
      return 0;
    }();
    Info.emplace_back(Val);
  }
  return Info;
}

static void updateInternalizationMD(Function *F, StringRef Kind,
                                    ArrayRef<size_t> LocalSizes) {
  MDNode *MD = F->getMetadata(SYCLInternalizer::Key);
  MDNode *LSMD = F->getMetadata(SYCLInternalizer::LocalSizeKey);
  assert(MD && LSMD && "Promotion metadata not available");
  assert(MD->getNumOperands() == LocalSizes.size() &&
         LSMD->getNumOperands() == LocalSizes.size() &&
         "Size mismatch in promotion metadata");
  for (auto I : enumerate(LocalSizes)) {
    const auto *CurMDS = cast<MDString>(MD->getOperand(I.index()));
    if (CurMDS->getString() == Kind) {
      if (I.value() == 0) {
        // The metadata indicates that this argument should be promoted, but the
        // analysis has deemed this infeasible (local size after analysis is 0).
        // Update the metadata-entry for this argument.
        auto *NewMDS = MDString::get(F->getContext(), NoPromotion);
        MD->replaceOperandWith(I.index(), NewMDS);
        auto *NewLS = MDString::get(F->getContext(), "");
        LSMD->replaceOperandWith(I.index(), NewLS);
      }
    }
  }
}

///
/// When performing internalization, GEP instructions must be remapped, as the
/// address space has changed from N to N / LocalSize.
static void remap(GetElementPtrInst *GEPI, std::size_t LocalSize) {
  IRBuilder<> Builder{GEPI};
  Value *C0 = Builder.getInt64(0);

  auto NIdx = GEPI->getNumIndices();
  if (NIdx > 1) {
    // `GEPI` indexes into an aggregate. If the first index is 0, the base
    // pointer is used as-is and we do not need to perform remapping. This is
    // the common case.
    // TODO: Support non-zero pointer offset, too. If the pointer operand is
    //       a GEP as well, we must check if the source element types match.
    assert(GEPI->idx_begin()->get() == C0);
    return;
  }

  if (LocalSize == 1) {
    // Squash the index and let instcombine clean-up afterwards.
    GEPI->idx_begin()->set(C0);
    return;
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
  Value *Remainder = Builder.CreateURem(Dividend, Builder.getInt64(LocalSize));
  GEPI->idx_begin()->set(Remainder);
}

///
/// Function to get the indices of a user in which a value appears.
static SmallVector<std::size_t>
getUsagesInternalization(const User *U, const Value *V, std::size_t LocalSize) {
  SmallVector<std::size_t> InternInfo;
  std::transform(U->op_begin(), U->op_end(), std::back_inserter(InternInfo),
                 [&](const Use &Us) { return Us == V ? LocalSize : 0; });
  return InternInfo;
}

Error SYCLInternalizerImpl::canPromoteCall(CallBase *C, const Value *Val,
                                           size_t LocalSize,
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

  SmallVector<size_t> InternInfo = getUsagesInternalization(C, Val, LocalSize);
  assert(!InternInfo.empty() && "Value must be used at least once");
  if (auto Err = checkArgsPromotable(C->getCalledFunction(), InternInfo)) {
    return Err;
  }
  return Error::success();
}

Error SYCLInternalizerImpl::canPromoteGEP(GetElementPtrInst *GEPI,
                                          const Value *Val, size_t LocalSize,
                                          bool InAggregate) const {
  if (cast<PointerType>(GEPI->getType())->getAddressSpace() == AS) {
    // If the GEPI is already using the correct address-space, no change is
    // required.
    return Error::success();
  }
  // Recurse to check all users of the GEP. We are either already in
  // `InAggregate` mode, or inspect the current instruction. Recall that a GEP's
  // first index is used to step through the base pointer, whereas any
  // additional indices represent addressing into an aggregrate type.
  return canPromoteValue(GEPI, LocalSize,
                         InAggregate || GEPI->getNumIndices() >= 2);
}

Error SYCLInternalizerImpl::canPromoteValue(Value *Val, size_t LocalSize,
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
              canPromoteCall(cast<CallBase>(I), Val, LocalSize, InAggregate)) {
        return Err;
      }
      break;
    case Instruction::GetElementPtr:
      if (auto Err = canPromoteGEP(cast<GetElementPtrInst>(I), Val, LocalSize,
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
    Function *F, SmallVectorImpl<size_t> &PromoteArgSizes) const {
  Error DeferredErrs = Error::success();
  for (auto I : enumerate(PromoteArgSizes)) {
    const size_t LocalSize = I.value();
    if (LocalSize == 0) {
      continue;
    }
    const size_t Index = I.index();
    Argument *Arg = F->getArg(Index);
    if (!isa<PointerType>(Arg->getType()) ||
        cast<Argument>(Arg)->hasByValAttr()) {
      // Omit non-pointer and byval arguments.
      PromoteArgSizes[Index] = 0;
      continue;
    }
    if (auto Err = canPromoteValue(Arg, LocalSize, /*InAggregate=*/false)) {
      // Set the local size to 0 to indicate that this argument should not be
      // promoted.
      PromoteArgSizes[Index] = 0;
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
                                       std::size_t LocalSize) const {

  const SmallVector<size_t> InternInfo =
      getUsagesInternalization(C, Val, LocalSize);
  assert(!InternInfo.empty() && "Value must be used at least once");
  Function *NewF = promoteFunctionArgs(C->getCalledFunction(), InternInfo,
                                       /* CreateAllocas */ false,
                                       /*KeepOriginal*/ true);

  C->setCalledFunction(NewF);
}

void SYCLInternalizerImpl::promoteGEPI(GetElementPtrInst *GEPI,
                                       const Value *Val, std::size_t LocalSize,
                                       bool InAggregate) const {
  // Not PointerType is unreachable. Other case is catched in caller.
  if (cast<PointerType>(GEPI->getType())->getAddressSpace() != AS) {
    if (!InAggregate)
      remap(GEPI, LocalSize);
    auto *ValTy = cast<PointerType>(Val->getType());
    GEPI->mutateType(PointerType::getWithSamePointeeType(
        cast<PointerType>(GEPI->getType()), ValTy->getAddressSpace()));
    // Recurse to promote to all users of the GEP. We are either already in
    // `InAggregate` mode, or inspect the current instruction. Recall that a
    // GEP's first index is used to step through the base pointer, whereas any
    // additional indices represent addressing into an aggregrate type.
    return promoteValue(GEPI, LocalSize,
                        InAggregate || GEPI->getNumIndices() >= 2);
  }
}

void SYCLInternalizerImpl::promoteValue(Value *Val, std::size_t LocalSize,
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
      promoteCall(cast<CallBase>(I), Val, LocalSize);
      break;
    case Instruction::GetElementPtr:
      promoteGEPI(cast<GetElementPtrInst>(I), Val, LocalSize, InAggregate);
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
static FunctionType *
getPromotedFunctionType(FunctionType *OrigTypes,
                        ArrayRef<std::size_t> PromoteToLocal, unsigned AS) {
  SmallVector<Type *> Types{OrigTypes->param_begin(), OrigTypes->param_end()};
  for (auto Arg : enumerate(PromoteToLocal)) {
    // No internalization.
    if (Arg.value() == 0) {
      continue;
    }
    Type *&Ty = Types[Arg.index()];
    // TODO: Catch this case earlier
    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      Ty = PointerType::getWithSamePointeeType(PtrTy, AS);
    }
  }
  return FunctionType::get(OrigTypes->getReturnType(), Types,
                           OrigTypes->isVarArg());
}

static Function *
getPromotedFunctionDeclaration(Function *F,
                               ArrayRef<std::size_t> PromoteToLocal,
                               unsigned AS, bool ChangeTypes) {
  FunctionType *Ty = F->getFunctionType();
  // If we do not need to change the types, we just copy the function
  // declaration.
  FunctionType *NewTy =
      ChangeTypes ? getPromotedFunctionType(Ty, PromoteToLocal, AS) : Ty;
  return Function::Create(NewTy, F->getLinkage(), F->getAddressSpace(),
                          F->getName(), F->getParent());
}

///
/// Determine the element type of a pointer value by inspecting its uses. This
/// is to determine the underlying type of a opaque pointer, whose element type
/// is determined by its uses.
static Type *getElementTypeFromUses(Value *PtrVal) {
  assert(PtrVal->getType()->isPointerTy() && "Not a pointer type");
  for (const auto &U : PtrVal->uses()) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      Type *InferredTy = nullptr;
      switch (I->getOpcode()) {
      case Instruction::Call:
      case Instruction::Invoke:
      case Instruction::CallBr: {
        auto *Call = cast<CallBase>(I);
        auto Index = Call->getArgOperandNo(&U);
        InferredTy =
            getElementTypeFromUses(Call->getCalledFunction()->getArg(Index));
        break;
      }
      case Instruction::AddrSpaceCast: {
        InferredTy = getElementTypeFromUses(I);
        break;
      }
      case Instruction::GetElementPtr:
        InferredTy = cast<GetElementPtrInst>(I)->getSourceElementType();
        break;
      case Instruction::Load:
        InferredTy = cast<LoadInst>(I)->getType();
        break;
      case Instruction::Store: {
        auto *Store = cast<StoreInst>(I);
        if (Store->getPointerOperand() == PtrVal) {
          InferredTy = Store->getValueOperand()->getType();
        }
        break;
      }
      default:
        // Unhandled case, rely on one of the following users to get type.
        break;
      }
      if (InferredTy) {
        return InferredTy;
      }
    }
  }
  return nullptr;
}

///
/// For private promotion, we want to replace each argument by an alloca.
Value *replaceByNewAlloca(Argument *Arg, unsigned AS, std::size_t LocalSize) {
  IRBuilder<> Builder{
      &*Arg->getParent()->getEntryBlock().getFirstInsertionPt()};
  auto *PtrTy = cast<PointerType>(Arg->getType());
  Type *Ty = getElementTypeFromUses(Arg);
  assert(Ty && "Could not determine pointer element type");
  auto *ArrTy = ArrayType::get(Ty, LocalSize);
  auto *Alloca = Builder.CreateAlloca(ArrTy, PtrTy->getAddressSpace());
  auto *Ptr = Builder.CreateInBoundsGEP(
      ArrTy, Alloca, {Builder.getInt64(0), Builder.getInt64(0)});
  Arg->replaceAllUsesWith(Ptr);
  Alloca->mutateType(PointerType::getWithSamePointeeType(
      cast<PointerType>(Alloca->getType()), AS));
  Ptr->mutateType(PointerType::getWithSamePointeeType(
      cast<PointerType>(Ptr->getType()), AS));
  return Ptr;
}

Function *SYCLInternalizerImpl::promoteFunctionArgs(
    Function *OldF, ArrayRef<std::size_t> PromoteToLocal, bool CreateAllocas,
    bool KeepOriginal) const {
  // We first declare the promoted function with the new signature.
  Function *NewF =
      getPromotedFunctionDeclaration(OldF, PromoteToLocal, AS,
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
  for (auto I : enumerate(PromoteToLocal)) {
    const auto LocalSize = I.value();
    if (LocalSize == 0) {
      continue;
    }
    const auto Index = I.index();
    Value *Arg{NewF->getArg(Index)};
    if (!isa<PointerType>(Arg->getType()) ||
        cast<Argument>(Arg)->hasByValAttr()) {
      // Omit non-pointer and byval arguments.
      continue;
    }
    if (CreateAllocas) {
      Arg = replaceByNewAlloca(cast<Argument>(Arg), AS, LocalSize);
    }
    promoteValue(Arg, LocalSize, /*InAggregate=*/false);
  }

  TargetInfo.updateAddressSpaceMetadata(NewF, PromoteToLocal, AS);

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
    Expected<SmallVector<size_t>> IndicesOrErr =
        getInternalizationFromMD(F, Kind);
    if (auto E = IndicesOrErr.takeError()) {
      handleAllErrors(std::move(E), [](const StringError &SE) {
        FUSION_DEBUG(llvm::dbgs() << SE.message() << "\n");
      });
      continue;
    }
    SmallVector<size_t> &Indices = *IndicesOrErr;

    // Analysis phase. Check which arguments, for which promotion was requested
    // by the user/runtime, can actually be promoted. The analysis will not
    // perform any modifications to the code. Instead, it sets the local size of
    // arguments, for which promotion would fail, to 0, so no promotion is
    // performned for this argument in the transformation phase.
    if (auto Err = checkArgsPromotable(F, Indices)) {
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
    updateInternalizationMD(F, Kind, Indices);

    if (llvm::none_of(Indices, [](size_t LS) { return LS > 0; })) {
      // If no arguments is requested & eligible for promotion after analysis,
      // skip the transformation phase.
      continue;
    }

    // Transformation phase. Promote the arguments for which it is actually safe
    // to perform promotion.
    promoteFunctionArgs(F, Indices, CreateAllocas);
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
    jit_compiler::ArgUsageMask NewArgInfo;
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
                {SYCLInternalizer::Key, SYCLInternalizer::LocalSizeKey});
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
