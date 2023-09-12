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

#include "transform/packetizer.h"

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/mangling.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <multi_llvm/creation_apis_helper.h>
#include <multi_llvm/llvm_version.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <memory>
#include <optional>

#include "analysis/instantiation_analysis.h"
#include "analysis/packetization_analysis.h"
#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "transform/instantiation_pass.h"
#include "transform/packetization_helpers.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz-packetization"

using namespace vecz;
using namespace llvm;

STATISTIC(VeczPacketized, "Number of instructions packetized [ID#P00]");
STATISTIC(VeczPacketizeFailCall,
          "Packetize: missing function declarations [ID#P81]");
STATISTIC(VeczPacketizeFailType,
          "Packetize: inconsistent vector parameters [ID#P87]");
STATISTIC(VeczPacketizeFailPtr,
          "Packetize: inconsistent pointer parameters [ID#P88]");
STATISTIC(VeczPacketizeFailStride,
          "Packetize: non-constant strides in pointer parameters [ID#P8A]");

// Just a little macro that can return an empty SmallVector, as a drop-in
// replacement for VECZ_FAIL_IF..
#define PACK_FAIL_IF(cond) \
  do {                     \
    if (cond) {            \
      return {};           \
    }                      \
  } while (false)

namespace {
// Returns a type equivalent to the input type plus padding.
// This converts a <3 x Ty> into a <4 x Ty>, leaving other types unchanged.
Type *getPaddedType(Type *Ty) {
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    if (VecTy->getNumElements() == 3) {
      return VectorType::get(VecTy->getElementType(),
                             ElementCount::getFixed(4));
    }
  }
  return Ty;
}

Type *getWideType(Type *Ty, ElementCount Factor) {
  unsigned Elts = 1;
  if (Ty->isVectorTy()) {
    auto *VecTy = cast<FixedVectorType>(Ty);
    Elts = VecTy->getNumElements();
    Ty = VecTy->getElementType();
  }
  return VectorType::get(Ty, Factor * Elts);
}
}  // namespace

using ValuePacket = SmallVector<Value *, 16>;

/// @brief Private implementation of the Packetizer.
/// It inherits its own outer class, which has only private constructors. This
/// allows us to pass it by reference to functions that need to access the
/// Packetizer, while also ensuring that a Packetizer cannot be created except
/// as the base class of its own implementation.
class Packetizer::Impl : public Packetizer {
 public:
  Impl(llvm::Function &F, llvm::FunctionAnalysisManager &AM, ElementCount Width,
       unsigned Dim);
  Impl() = delete;
  Impl(const Packetizer &) = delete;
  Impl(Packetizer &&) = delete;
  ~Impl();

  bool packetize();

  /// @brief Handle packetization failure. This method ensures that
  /// packetization failure does not leave behind invalid IR.
  void onFailure();

  /// @brief Packetize the given value from the function.
  ///
  /// @param[in] V Value to packetize.
  ///
  /// @return Packetized value.
  Result packetize(Value *V);

  /// @brief Packetize the given value and return the packet by values
  ///
  /// @param[in] V Value to packetize.
  ///
  /// @return Packetized values.
  ValuePacket packetizeAndGet(Value *V);

  /// @brief Packetize the given value to a specified packet width, and return
  /// the packet by values
  ///
  /// @param[in] V Value to packetize.
  /// @param[in] Width the requested packet width
  ///
  /// @return Packetized values.
  ValuePacket packetizeAndGet(Value *V, unsigned Width);

  /// @brief Packetize the given value from the function, only if it is a
  /// varying value. Ensures Mask Varying values are handled correctly.
  ///
  /// @param[in] V Value to packetize.
  ///
  /// @return Packetized value if varying, or the original value if Uniform.
  Value *packetizeIfVarying(Value *V);

  /// @brief Packetize a uniform value by broadcasting to all vector lanes.
  ///
  /// @param[in] V Value to broadcast
  ///
  /// @return Packetized instruction
  Result broadcast(Value *V);
  /// @brief Reduce a varying boolean condition to a scalar
  ///
  /// @param[in] cond Condition to packetize.
  /// @param[in] terminator Terminator instruction.
  /// @param[in] allOf Whether to create a all of mask, or any of.
  ///
  /// @return reduced boolean value.
  Value *reduceBranchCond(Value *cond, Instruction *terminator, bool allOf);
  /// @brief Compute the ideal packet width for subwidening the given type
  ///
  /// @param[in] ty Type of the value to subwiden
  /// @param[in] limit The maximum vector width we allow
  ///
  /// @return width of the packet to create
  unsigned getPacketWidthForType(Type *ty, unsigned limit = ~0u) const;
  /// @brief Packetize an instruction.
  ///
  /// @param[in] Ins Instruction to packetize.
  ///
  /// @return Packetized instructions.
  Result packetizeInstruction(Instruction *Ins);
  /// @brief Packetize a mask-varying instruction.
  ///
  /// @param[in] I Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *packetizeMaskVarying(Instruction *I);
  /// @brief Packetize a mask-varying subgroup/workgroup reduction.
  ///
  /// @param[in] I Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *packetizeGroupReduction(Instruction *I);
  /// @brief Packetize a subgroup/workgroup broadcast.
  ///
  /// @param[in] I Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *packetizeGroupBroadcast(Instruction *I);
  /// @brief Returns true if the instruction is a subgroup shuffle.
  ///
  /// @param[in] I Instruction to query.
  ///
  /// @return True if the instruction is a call to a mux subgroup shuffle
  /// builtin.
  bool isSubgroupShuffle(Instruction *I);
  /// @brief Packetize PHI node.
  ///
  /// @param[in] Phi PHI Node to packetize.
  ///
  /// @return Packetized values.
  ValuePacket packetizePHI(PHINode *Phi);
  /// @brief Packetize a call instruction.
  ///
  /// @param[in] CI Call Instruction to packetize.
  ///
  /// @return Packetized values.
  ValuePacket packetizeCall(CallInst *CI);
  /// @brief Packetize a subgroup scan.
  ///
  /// @param[in] CI CallInst to packetize.
  /// @param[in] Scan type of subgroup scan to packetized.
  ///
  /// @return Packetized values.
  ValuePacket packetizeSubgroupScan(CallInst *CI,
                                    compiler::utils::GroupCollective Scan);
  /// @brief Perform post-packetization tasks for the given scalar value.
  ///
  /// @param[in] Scalar Scalar value to assign a vectorized value.
  /// @param[in] Vectorized Packetized value to assign.
  ///
  /// @return Packetized values.
  Result assign(Value *Scalar, Value *Vectorized);
  /// @brief Vectorize an instruction.
  ///
  /// @param[in] Ins Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *vectorizeInstruction(Instruction *Ins);
  /// @brief Packetize a load instruction.
  ///
  /// @param[in] Load Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeLoad(LoadInst *Load);
  /// @brief Packetize a store instruction.
  ///
  /// @param[in] Store Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeStore(StoreInst *Store);
  /// @brief Packetize a memory operation.
  ///
  /// @param[in] Op Memory operation to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeMemOp(MemOp &Op);
  /// @brief Packetize a GEP instruction.
  ///
  /// @param[in] GEP Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeGEP(GetElementPtrInst *GEP);
  /// @brief Packetize a cast instruction.
  ///
  /// @param[in] CastI Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeCast(CastInst *CastI);
  /// @brief Packetize a binary operator instruction.
  ///
  /// @param[in] BinOp Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeBinaryOp(BinaryOperator *BinOp);
  /// @brief Packetize a freeze instruction.
  ///
  /// @param[in] FreezeI Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeFreeze(FreezeInst *FreezeI);
  /// @brief Packetize a unary operator instruction.
  ///
  /// @param[in] UnOp Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeUnaryOp(UnaryOperator *UnOp);
  /// @brief Packetize an integer compare instruction.
  ///
  /// @param[in] Cmp Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeICmp(ICmpInst *Cmp);
  /// @brief Packetize a floating-point compare instruction.
  ///
  /// @param[in] Cmp Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeFCmp(FCmpInst *Cmp);
  /// @brief Packetize a select instruction.
  ///
  /// @param[in] Select Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeSelect(SelectInst *Select);
  /// @brief Packetize a return instruction.
  ///
  /// @param[in] Return Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *vectorizeReturn(ReturnInst *Return);
  /// @brief Packetize a call instruction.
  ///
  /// @param[in] CI Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *vectorizeCall(CallInst *CI);
  /// @brief Packetize a call to a work-group builtin.
  ///
  /// @param[in] CI Instruction to packetize.
  /// @param[in] Builtin Builtin identifier.
  ///
  /// @return Packetized instruction.
  Value *vectorizeWorkGroupCall(CallInst *CI,
                                compiler::utils::BuiltinCall const &Builtin);
  /// @brief Packetize an alloca instruction.
  ///
  /// @param[in] Alloca Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *vectorizeAlloca(AllocaInst *Alloca);
  /// @brief Packetize an extract value instruction.
  ///
  /// @param[in] ExtractElement Instruction to packetize.
  ///
  /// @return Packetized instruction.
  Value *vectorizeExtractValue(ExtractValueInst *ExtractElement);
  /// @brief Packetize an insert element instruction.
  ///
  /// @param[in] InsertElement Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeInsertElement(InsertElementInst *InsertElement);
  /// @brief Packetize an extract element instruction.
  ///
  /// @param[in] ExtractElement Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeExtractElement(ExtractElementInst *ExtractElement);
  /// @brief Packetize a shuffle vector instruction.
  ///
  /// @param[in] Shuffle Instruction to packetize.
  ///
  /// @return Packetized instruction.
  ValuePacket packetizeShuffleVector(ShuffleVectorInst *Shuffle);
  /// @brief Preserves debug information attached to old scalar instruction,
  ///        updating the debug info type to match the vector width.
  ///
  /// @param[in] Scalar Scalar instruction before packetization.
  /// @param[in] Packet Packetized instruction.
  void vectorizeDI(Instruction *Scalar, Value *Packet);

  /// @brief Helps handle instructions that cannot be packetized.
  std::unique_ptr<InstantiationPass> Instantiator;

  /// @brief List of phi nodes that can be used by passes to defer the
  /// processing of these nodes.
  std::vector<PHINode *> pendingPhis;

  /// @brief The target transform info
  const TargetTransformInfo TTI;
};

Packetizer::Packetizer(llvm::Function &F, llvm::FunctionAnalysisManager &AM,
                       ElementCount Width, unsigned Dim)
    : AM(AM),
      VU(AM.getResult<VectorizationUnitAnalysis>(F).getVU()),
      Ctx(AM.getResult<VectorizationContextAnalysis>(F).getContext()),
      Choices(VU.choices()),
      UVR(AM.getResult<UniformValueAnalysis>(F)),
      SAR(AM.getResult<StrideAnalysis>(F)),
      PAR(AM.getResult<PacketizationAnalysis>(F)),
      F(F),
      SimdWidth(Width),
      Dimension(Dim) {}

Packetizer::Impl::Impl(llvm::Function &F, llvm::FunctionAnalysisManager &AM,
                       ElementCount Width, unsigned Dim)
    : Packetizer(F, AM, Width, Dim), TTI(Ctx.getTargetTransformInfo(F)) {
  Instantiator.reset(new InstantiationPass(*this));
}

Packetizer::Impl::~Impl() = default;

bool Packetizer::packetize(llvm::Function &F, llvm::FunctionAnalysisManager &AM,
                           ElementCount Width, unsigned Dim) {
  Impl impl(F, AM, Width, Dim);
  bool Res = impl.packetize();
  if (!Res) {
    impl.onFailure();
  }
  return Res;
}

bool Packetizer::Impl::packetize() {
  LLVM_DEBUG(if (PAR.isEmpty()) {
    llvm::dbgs() << "No vector leaves in function "
                 << VU.scalarFunction()->getName() << "\n";
  });

  // If requested, set up the base vector length for this kernel based on the
  // number of remaining work items: the local size minus the local id. Since
  // VP intrinsics are undefined for %evl values larger than the actual vector
  // width, we also constrain it based on the vectorization width.
  BasicBlock &EntryBB = F.getEntryBlock();
  IRBuilder<> B(&*EntryBB.getFirstInsertionPt());

  if (Choices.vectorPredication()) {
    auto &M = *F.getParent();
    auto *const I32Ty = Type::getInt32Ty(F.getContext());
    auto *const LocalIdFn = Ctx.builtins().getOrDeclareMuxBuiltin(
        compiler::utils::eMuxBuiltinGetLocalId, M);
    auto *const LocalSizeFn = Ctx.builtins().getOrDeclareMuxBuiltin(
        compiler::utils::eMuxBuiltinGetLocalSize, M);
    assert(LocalIdFn && LocalSizeFn && "Unable to create mux builtins");
    auto *const ID =
        B.CreateCall(LocalIdFn, B.getInt32(VU.dimension()), "local.id");
    ID->setAttributes(LocalIdFn->getAttributes());
    ID->setCallingConv(LocalIdFn->getCallingConv());
    auto *const Size =
        B.CreateCall(LocalSizeFn, B.getInt32(VU.dimension()), "local.size");
    Size->setAttributes(LocalSizeFn->getAttributes());
    Size->setCallingConv(LocalSizeFn->getCallingConv());
    VECZ_FAIL_IF(!ID || !Size);

    VL = B.CreateSub(Size, ID, "work.remaining", /*HasNUW*/ true,
                     /*HasNSW*/ true);

    if (auto *RVVVL = Ctx.targetInfo().createVPKernelWidth(
            B, VL, /*WidestType*/ 32, VU.width())) {
      VL = RVVVL;
    } else {
      auto *const Scaling =
          ConstantInt::get(VL->getType(), VU.width().getKnownMinValue());
      auto *const VectorLength =
          VU.width().isScalable() ? B.CreateVScale(Scaling) : Scaling;
      VL = B.CreateIntrinsic(Intrinsic::umin, {VL->getType()},
                             {VL, VectorLength});

      VL = B.CreateTrunc(VL, I32Ty);
    }
  }

  // Manifest the memory operation stride values as actual `llvm::Value`s
  SAR.manifestAll(B);

  // Pre-process the arguments first to replace any placeholders with their
  // proper vector values, and convert pointer return arguments to vector of
  // pointers where required.
  {
    Value *idxVector = nullptr;
    for (const auto &TargetArg : VU.arguments()) {
      if (auto *const Placeholder = TargetArg.Placeholder) {
        auto &info = packets[Placeholder];
        info.vector = TargetArg.NewArg;
        info.numInstances = 1;
      } else if (TargetArg.PointerRetPointeeTy &&
                 PAR.needsPacketization(TargetArg.NewArg)) {
        if (!idxVector) {
          idxVector = multi_llvm::createIndexSequence(
              B, VectorType::get(B.getInt32Ty(), SimdWidth), SimdWidth,
              "index.vec");
        }

        // CA-3943 this implementation looks unlikely to be correct, but for
        // now we just maintain the original behaviour, until we have a better
        // idea of what is going on or whether any of this is still needed.
        // This case will never be encountered during kernel vectorization.
        auto *const Arg = TargetArg.NewArg;
        auto *const EleTy = TargetArg.PointerRetPointeeTy;
        auto &info = packets[Arg];
        info.vector = B.CreateGEP(EleTy, Arg, idxVector);
        info.numInstances = 1;
      }
    }
  }

  // Build an ordered list of the instructions to packetize, in depth first
  // order so that we don't have to recurse too much. We build the list first
  // because packetization of calls can produce loops, which messes up our
  // iteration over the basic blocks of the function.
  std::vector<Instruction *> ordered;
  for (auto *BB : depth_first(&F)) {
    for (auto &I : *BB) {
      if (PAR.needsPacketization(&I)) {
        ordered.push_back(&I);
      }
    }
  }

  for (auto *const I : ordered) {
    if (!packetize(I)) {
      emitVeczRemarkMissed(&F, I, "Could not packetize");
      VECZ_FAIL();
    }
  }

  // Packetize remaining phi nodes until they have all been packetized.
  // Packetizing one phi node may involve the packetization of another node.
  // Some nodes might need to be instantiated instead of being packetized, but
  // we are handling this here because the instantiation pass is not run as a
  // standalone pass.
  // Note: pendingPhis *may* change as we progress through this loop, by
  // calling packetize(Incoming). Therefore we can't cache the vector size when
  // setting up the loop.
  for (unsigned i = 0; i < pendingPhis.size(); i++) {
    PHINode *Phi = pendingPhis[i];
    auto &info = packets[Phi];
    assert(info.numInstances > 0 && "A PHI pending packetization has no stub");
    if (info.numInstances == 1) {
      auto *NewPhi = cast<PHINode>(info.vector);
      for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
        Value *Incoming = Phi->getIncomingValue(i);
        BasicBlock *BB = Phi->getIncomingBlock(i);
        Value *VecIncoming = packetize(Incoming).getAsValue();
        VECZ_FAIL_IF(!VecIncoming);
        NewPhi->addIncoming(VecIncoming, BB);
      }
    } else {
      const auto PhiPacket = info.getRange(packetData);
      for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
        Value *Incoming = Phi->getIncomingValue(i);
        BasicBlock *BB = Phi->getIncomingBlock(i);
        auto PackIncoming = packetize(Incoming).getAsPacket(PhiPacket.size());
        for (unsigned j = 0; j < PhiPacket.size(); ++j) {
          auto *NewPhi = cast<PHINode>(PhiPacket.at(j));
          auto *Incoming = PackIncoming.at(j);
          VECZ_FAIL_IF(!NewPhi);
          VECZ_FAIL_IF(!Incoming);
          NewPhi->addIncoming(Incoming, BB);
        }
      }
    }
    IC.deleteInstructionLater(Phi);
  }

  auto *insertPt = &*EntryBB.begin();
  for (auto &I : EntryBB) {
    auto *const alloca = dyn_cast<AllocaInst>(&I);
    if (!alloca) {
      insertPt = I.getNextNonDebugInstruction();
      continue;
    }

    while (isa<AllocaInst>(insertPt)) {
      insertPt = insertPt->getNextNonDebugInstruction();
    }

    // It's possible for some uses of the alloca to be packetized and others
    // not. For instance, where we have a store to a constant address, since
    // the execution order of work items is undefined, the data operand need
    // not be packetized, and we can end up with uses of the scalar alloca
    // still present in the vector function. In such a case we can replace it
    // with the first element of the packetized alloca.
    if (auto res = getPacketized(alloca)) {
      SmallVector<Value *, 16> vals;
      res.getPacketValues(vals);
      if (vals.empty()) {
        // It is a broadcast value, so we don't need to do anything.
        continue;
      }
      auto *element0 = vals.front();

      if (!isa<AllocaInst>(element0)) {
        assert(isa<GetElementPtrInst>(element0) && "vecz: expected GEP");
        auto *const GEP = cast<GetElementPtrInst>(element0);
        // If the alloca was packetized, it will be indexed by a GEP.
        // We only need the original, un-indexed pointer.
        alloca->replaceAllUsesWith(GEP->getPointerOperand());
        continue;
      }

      if (element0->getType()->isVectorTy()) {
        B.SetInsertPoint(insertPt);
        element0 = B.CreateExtractElement(element0, B.getInt32(0));
      }
      alloca->replaceAllUsesWith(element0);
      continue;
    }

    // We have to widen allocas if they are varying, regardless of the result
    // of the packetization analysis, because they need enough storage for all
    // lanes, even though they are only accessed through a scalar pointer.
    // We do this last, otherwise it messes with the stride analysis etc.
    // Only non-instantiated allocas should be left by now.
    if (!UVR.isVarying(alloca)) {
      continue;
    }
    // Array allocas need to be instantiated.
    assert(!alloca->isArrayAllocation() &&
           "vecz: unexpected array alloca; should have been instantiated");

    B.SetInsertPoint(alloca);
    auto *const dataTy = alloca->getAllocatedType();
    if (dataTy->isVectorTy() || VectorType::isValidElementType(dataTy)) {
      // We can vectorize or vector widen this type.
      auto *const newAlloca =
          B.CreateAlloca(getWideType(getPaddedType(dataTy), SimdWidth));
      newAlloca->setAlignment(alloca->getAlign());
      newAlloca->takeName(alloca);

      // Absorb other bitcasts (e.g. i8* for lifetime instrinsics, or bitcasts
      // back to vector type for contiguous loads/stores)
      bool needCast = false;
      auto *const newTy = newAlloca->getType();
      for (Use &U : alloca->uses()) {
        auto *const user = dyn_cast<BitCastInst>(U.getUser());
        if (!user) {
          needCast = true;
          continue;
        }

        auto *const dstTy = user->getType();
        if (dstTy == newTy) {
          // Bitcasts totally redundant
          user->replaceAllUsesWith(newAlloca);
        } else {
          // Bitcast into different bitcast
          B.SetInsertPoint(user);
          user->replaceAllUsesWith(B.CreateBitCast(newAlloca, user->getType()));
        }
        IC.deleteInstructionLater(cast<Instruction>(user));
      }

      if (needCast) {
        // Insert the bitcast after all the allocas
        B.SetInsertPoint(insertPt);
        auto *const scalarPtr =
            B.CreatePointerCast(newAlloca, alloca->getType());
        alloca->replaceAllUsesWith(scalarPtr);
      }
    } else {
      // We couldn't vectorize the type, so create an array instead.
      VECZ_FAIL_IF(SimdWidth.isScalable());
      unsigned const fixedWidth = SimdWidth.getFixedValue();

      AllocaInst *const wideAlloca =
          B.CreateAlloca(dataTy, getSizeInt(B, fixedWidth), alloca->getName());
      auto align = alloca->getAlign();

      // Make sure the alloca has an alignment at least as wide as any of the
      // packetized loads or stores using it.
      SmallVector<Instruction *, 8> users;
      for (Use &U : alloca->uses()) {
        users.push_back(cast<Instruction>(U.getUser()));
      }
      while (!users.empty()) {
        auto *const user = users.pop_back_val();
        if (isa<BitCastInst>(user) || isa<GetElementPtrInst>(user)) {
          for (Use &U : user->uses()) {
            users.push_back(cast<Instruction>(U.getUser()));
          }
        } else if (auto memop = MemOp::get(user)) {
          auto const memAlign = memop->getAlignment();
          if (memAlign > align.value()) {
            align = Align(memAlign);
          }
        }
      }

      wideAlloca->setAlignment(align);
      wideAlloca->takeName(alloca);

      // It's just a direct replacement.
      alloca->replaceAllUsesWith(wideAlloca);
    }

    // Note that we don't assign the widened allocas a packet, because they
    // are not really being packetized. The problem is, a packetized alloca
    // would be expected to be a vector of pointers to scalars, not a scalar
    // pointer to a vector. Only instantiation can create such a packet.
    IC.deleteInstructionLater(alloca);
  }

  compiler::utils::NameMangler Mangler(&F.getContext());

  // Handle __mux_get_sub_group_size specially (i.e., not in BuiltinInfo) since
  // inlining it requires extra vectorization context, such as the vectorization
  // width and choices; this inlining is too tightly coupled to the vectorizer
  // context to exist in a generic sense.
  for (auto &BB : F) {
    for (auto &I : BB) {
      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI) {
        continue;
      }

      if (auto *const Callee = CI->getCalledFunction();
          Callee && Ctx.builtins().analyzeBuiltin(*Callee).ID ==
                        compiler::utils::eMuxBuiltinGetSubGroupSize) {
        auto *const replacement = [this](CallInst *CI) -> Value * {
          // The vectorized sub-group size is the mux sub-group reduction sum
          // of all of the vectorized sub-group sizes:
          // |   mux 0     |      mux 1       |
          // | < a,b,c,d > | < e,f,g > (vl=3) |
          // The total sub-group size above is 4 + 3 => 7.
          // Note that this expects that the mux sub-group consists entirely of
          // equivalently vectorized kernels.
          Value *VecgroupSize;
          IRBuilder<> B(CI);
          auto *const I32Ty = B.getInt32Ty();
          if (VL) {
            VecgroupSize = VL;
          } else {
            auto *const VFVal = B.getInt32(SimdWidth.getKnownMinValue());
            if (!SimdWidth.isScalable()) {
              VecgroupSize = VFVal;
            } else {
              VecgroupSize = B.CreateVScale(VFVal);
            }
          }
          assert(VecgroupSize && "Could not determine vector group size");

          auto *ReduceFn = Ctx.builtins().getOrDeclareMuxBuiltin(
              compiler::utils::eMuxBuiltinSubgroupReduceAdd, *F.getParent(),
              {I32Ty});
          assert(ReduceFn && "Could not get reduction builtin");

          return B.CreateCall(ReduceFn, VecgroupSize, "subgroup.size");
        }(CI);
        CI->replaceAllUsesWith(replacement);
        IC.deleteInstructionLater(CI);
      }
    }
  }

  IC.deleteInstructions();
  return true;
}

void Packetizer::Impl::onFailure() {
  // On failure, clean up pending Phis, which may still be invalid in that they
  // have no incoming operands. For simplicity, just erase and replace all of
  // them with undef: the failed vectorized function will be removed anyway.
  for (auto *Phi : pendingPhis) {
    auto &info = packets[Phi];
    assert(info.numInstances > 0 && "A PHI pending packetization has no stub");
    if (info.numInstances == 1) {
      IRCleanup::deleteInstructionNow(cast<PHINode>(info.vector));
    } else {
      const auto PhiPacket = info.getRange(packetData);
      for (unsigned j = 0; j < PhiPacket.size(); ++j) {
        IRCleanup::deleteInstructionNow(cast<PHINode>(PhiPacket.at(j)));
      }
    }
  }
}

Packetizer::Result Packetizer::packetize(Value *V) {
  // This is safe because we only ever create an instance of Impl, never an
  // instance of the base class.
  return static_cast<Impl *>(this)->packetize(V);
}

Packetizer::Result Packetizer::getPacketized(Value *V) {
  auto found = packets.find(V);
  auto *info = found != packets.end() ? &found->second : nullptr;
  return Packetizer::Result(*this, V, info);
}

PacketRange Packetizer::createPacket(Value *V, unsigned width) {
  auto &info = packets[V];
  info.numInstances = width;
  return Result(*this, V, &info).createPacket(width);
}

Value *Packetizer::Impl::reduceBranchCond(Value *cond, Instruction *terminator,
                                          bool allOf) {
  // Get the branch condition at its natural packet width
  auto conds = packetizeAndGet(cond);
  VECZ_FAIL_IF(conds.empty());

  // Branches can only take a scalar mask. The new branch condition is true
  // only if the original condition is true for any lane (or for all lanes if
  // the condition is used in a BOSCC block indirection.)
  IRBuilder<> B(terminator);
  auto const name = cond->getName();

  // Reduce the packet to a single value
  auto w = conds.size();

  if (VL && w != 1) {
    emitVeczRemarkMissed(&F, cond,
                         "Can not vector-predicate packets larger than 1");
    return nullptr;
  }

  while ((w >>= 1)) {
    for (decltype(w) i = 0; i < w; ++i) {
      conds[i] =
          allOf ? B.CreateAnd(conds[i], conds[i + w], Twine(name, ".all_of"))
                : B.CreateOr(conds[i], conds[i + w], Twine(name, ".any_of"));
    }
  }

  RecurKind kind = allOf ? RecurKind::And : RecurKind::Or;

  // VP reduction intrinsics didn't make it into LLVM 13 so we have to make do
  // by pre-sanitizing the input such that elements past VL get the identity
  // value.
  Value *&f = conds.front();

  if (VL) {
    f = sanitizeVPReductionInput(B, f, VL, kind);
    VECZ_FAIL_IF(!f);
  }

  return createSimpleTargetReduction(B, &TTI, f, kind);
}

Packetizer::Result Packetizer::Impl::assign(Value *Scalar, Value *Vectorized) {
  if (!Vectorized) {
    emitVeczRemarkMissed(&F, Scalar, "Failed to vectorize");
    return Packetizer::Result(*this);
  } else {
    ++VeczPacketized;
    auto &info = packets[Scalar];
    info.vector = Vectorized;
    info.numInstances = 1;
    return Packetizer::Result(*this, Scalar, &info);
  }
}

Value *Packetizer::Impl::packetizeIfVarying(Value *V) {
  if (UVR.isVarying(V)) {
    return packetize(V).getAsValue();
  } else if (UVR.isMaskVarying(V)) {
    VECZ_FAIL_IF(!packetize(V));
  }
  return V;
}

Packetizer::Result Packetizer::Impl::packetize(Value *V) {
  // Do not packetize the same value twice.
  if (const auto res = getPacketized(V)) {
    return res;
  }
  // Now check whether this value is actually packetizable.
  if (!Ctx.targetInfo().canPacketize(V, SimdWidth)) {
    return Packetizer::Result(*this);
  }

  if (!isa<Instruction>(V)) {
    return broadcast(V);
  }

  auto *const Ins = cast<Instruction>(V);

  // FIXME: Add support for vectorizing sub-group shuffles
  if (isSubgroupShuffle(Ins)) {
    emitVeczRemarkMissed(&F, Ins, "Could not packetize sub-group shuffle");
    return Packetizer::Result(*this);
  }

  if (auto *const Branch = dyn_cast<BranchInst>(Ins)) {
    if (Branch->isConditional()) {
      // varying reductions need to be packetized
      auto *newCond = packetize(Branch->getCondition()).getAsValue();
      if (!newCond) {
        return Packetizer::Result(*this);
      }

      // Packetization should normally have produced a reduction to scalar.
      // However, when Packetize Uniform is on, a uniform branch won't have
      // a divergence reduction so it will need reducing manually here.
      if (newCond->getType()->isVectorTy()) {
        IRBuilder<> B(Branch);
        RecurKind kind = RecurKind::Or;
        // Sanitize VP reduction inputs, if required.
        if (VL) {
          newCond = sanitizeVPReductionInput(B, newCond, VL, kind);
          if (!newCond) {
            return Packetizer::Result(*this);
          }
        }
        newCond = createSimpleTargetReduction(B, &TTI, newCond, kind);
      }

      Branch->setCondition(newCond);
    }
    return broadcast(Ins);
  }

  if (isa<SwitchInst>(Ins)) {
    // we can't handle varying switches
    return Packetizer::Result(*this);
  }

  if (UVR.isMaskVarying(Ins)) {
    if (auto *const res = packetizeMaskVarying(Ins)) {
      return broadcast(res);
    }
    // Fall back on instantiation if the instruction could not be packetized
    Instantiator->instantiate(Ins);
    return getPacketized(Ins);
  }

  if (auto *reduction = packetizeGroupReduction(Ins)) {
    return broadcast(reduction);
  }

  if (auto *brdcast = packetizeGroupBroadcast(Ins)) {
    return broadcast(brdcast);
  }

  // Check if we should broadcast the instruction.
  // Broadcast uniform instructions, unless we want to packetize uniform
  // instructions as well. We can assume that isMaskVarying is false at this
  // point.
  bool shouldBroadcast = !UVR.isVarying(Ins) && !Choices.packetizeUniform();
  // Or unless this instruction is in a loop and we want to packetize uniform
  // instructions in loops
  if (shouldBroadcast && Choices.packetizeUniformInLoops()) {
    const LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
    shouldBroadcast = !LI.getLoopFor(Ins->getParent());
  }

  // The packetization of a mask-varying value takes care of its own broadcast
  if (shouldBroadcast) {
    // Insert broadcast instructions after the instruction to broadcast
    return broadcast(Ins);
  }

  if (const auto res = packetizeInstruction(Ins)) {
    return res;
  }
  // Fall back on instantiation if the instruction could not be packetized,
  // unless we're vector predicating.
  if (VL) {
    return Packetizer::Result(*this);
  }
  Instantiator->instantiate(Ins);
  return getPacketized(Ins);
}

ValuePacket Packetizer::Impl::packetizeAndGet(Value *v) {
  ValuePacket results;
  if (auto res = packetize(v)) {
    res.getPacketValues(results);
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeAndGet(Value *v, unsigned w) {
  ValuePacket results;
  if (auto res = packetize(v)) {
    res.getPacketValues(w, results);
  }
  return results;
}

Packetizer::Result Packetizer::Impl::broadcast(Value *V) {
  return Result(*this, V, &packets[V]);
}

unsigned Packetizer::Impl::getPacketWidthForType(Type *ty,
                                                 unsigned limit) const {
  if (SimdWidth.isScalable()) {
    return 1;
  }

  const unsigned simdWidth = SimdWidth.getFixedValue();
  unsigned maxWidth = 0;

  if (!Choices.targetIndependentPacketization()) {
    maxWidth = std::min(limit, Ctx.targetInfo().getVectorWidthForType(
                                   TTI, *ty->getScalarType()));

    // We let the target return a value wider than the SIMD Width, but not
    // narrower.
    if (maxWidth) {
      maxWidth = std::max(simdWidth, maxWidth);
    }
  }

  if (maxWidth == 0) {
    maxWidth = std::max(simdWidth, 16u);
  }

  unsigned elts = 1;
  if (ty->isVectorTy()) {
    auto *vecTy = cast<FixedVectorType>(ty);
    elts = vecTy->getNumElements();
  }

  const unsigned fullWidth = elts * simdWidth;
  if (fullWidth <= maxWidth) {
    return 1;
  }

  // Round up to the next power of two..
  // This should only be needed if the type was a 3-vector..
  // Note that we don't really expect huge values here, over 16 is still
  // currently not officially supported, over 256 would be astonishing,
  // and over 65536 would be inconcievable, so we don't bother to >> 16.
  unsigned width = fullWidth / maxWidth - 1;
  width |= width >> 1;
  width |= width >> 2;
  width |= width >> 4;
  width |= width >> 8;

  // Can't have a packet wider than the simdWidth..
  return std::min(width + 1, simdWidth);
}

Packetizer::Result Packetizer::Impl::packetizeInstruction(Instruction *Ins) {
  ValuePacket results;

  // Figure out what kind of instruction it is and try to vectorize it.
  switch (Ins->getOpcode()) {
    default:
      if (Ins->isBinaryOp()) {
        results = packetizeBinaryOp(cast<BinaryOperator>(Ins));
      } else if (Ins->isCast()) {
        results = packetizeCast(cast<CastInst>(Ins));
      } else if (Ins->isUnaryOp()) {
        results = packetizeUnaryOp(cast<UnaryOperator>(Ins));
      }
      break;

    case Instruction::PHI:
      results = packetizePHI(cast<PHINode>(Ins));
      break;
    case Instruction::GetElementPtr:
      results = packetizeGEP(cast<GetElementPtrInst>(Ins));
      break;
    case Instruction::Store:
      results = packetizeStore(cast<StoreInst>(Ins));
      break;
    case Instruction::Load:
      results = packetizeLoad(cast<LoadInst>(Ins));
      break;
    case Instruction::Call:
      results = packetizeCall(cast<CallInst>(Ins));
      break;
    case Instruction::ICmp:
      results = packetizeICmp(cast<ICmpInst>(Ins));
      break;
    case Instruction::FCmp:
      results = packetizeFCmp(cast<FCmpInst>(Ins));
      break;
    case Instruction::Select:
      results = packetizeSelect(cast<SelectInst>(Ins));
      break;
    case Instruction::InsertElement:
      results = packetizeInsertElement(cast<InsertElementInst>(Ins));
      break;
    case Instruction::ExtractElement:
      results = packetizeExtractElement(cast<ExtractElementInst>(Ins));
      break;
    case Instruction::ShuffleVector:
      results = packetizeShuffleVector(cast<ShuffleVectorInst>(Ins));
      break;
    case Instruction::Freeze:
      results = packetizeFreeze(cast<FreezeInst>(Ins));
      break;
  }

  if (!results.empty()) {
    auto packetWidth = results.size();
    if (packetWidth == 1) {
      Value *vec = results.front();
      if (vec != Ins) {
        // Only delete if the vectorized value is different from the scalar.
        IC.deleteInstructionLater(Ins);
      }
      vectorizeDI(Ins, vec);
      return assign(Ins, vec);
    } else {
      IC.deleteInstructionLater(Ins);
      auto &info = packets[Ins];
      auto res = Result(*this, Ins, &info);
      auto P = res.createPacket(packetWidth);
      for (unsigned i = 0; i < packetWidth; ++i) {
        P[i] = results[i];
        // TODO CA-3376: vectorize the debug instructions
      }
      info.numInstances = packetWidth;
      ++VeczPacketized;
      return res;
    }
  }

  if (auto *vec = vectorizeInstruction(Ins)) {
    return assign(Ins, vec);
  }

  return Packetizer::Result(*this, Ins, nullptr);
}

Value *Packetizer::Impl::packetizeGroupReduction(Instruction *I) {
  auto *const CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->getCalledFunction()) {
    return nullptr;
  }
  compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  Function *callee = CI->getCalledFunction();

  auto const Builtin = BI.analyzeBuiltin(*callee);
  auto const Info = BI.isMuxGroupCollective(Builtin.ID);

  if (!Info || (!Info->isSubGroupScope() && !Info->isWorkGroupScope()) ||
      (!Info->isAnyAll() && !Info->isReduction())) {
    return nullptr;
  }

  bool isWorkGroup = Info->isWorkGroupScope();
  unsigned argIdx = isWorkGroup ? 1 : 0;

  SmallVector<Value *, 16> opPackets;
  IRBuilder<> B(CI);
  auto *const argTy = CI->getArgOperand(argIdx)->getType();
  auto packetWidth = getPacketWidthForType(argTy);

  // Don't vector predicate if we have to split into multiple packets. The
  // introduction of instructions to manage the splitting up of our VL into N
  // chunks is likely to kill performance anyway.
  if (VL && packetWidth != 1) {
    emitVeczRemarkMissed(&F, CI,
                         "Can not vector-predicate packets larger than 1");
    return nullptr;
  }

  auto op = packetize(CI->getArgOperand(argIdx));

  // Reduce the packet values in-place.
  // TODO: can we add 'reassoc' to the floating-point reductions to absolve
  // them of ordering? See CA-3969.
  op.getPacketValues(packetWidth, opPackets);

  // When in VP mode, pre-sanitize the reduction input (before VP reduction
  // intrinsics, introduced in LLVM 14)
  if (VL) {
    assert(opPackets.size() == 1 &&
           "Should have bailed if dealing with more than one packet");
    Value *&val = opPackets.front();
    val = sanitizeVPReductionInput(B, val, VL, Info->Recurrence);
    if (!val) {
      emitVeczRemarkMissed(
          &F, CI, "Can not vector-predicate workgroup/subgroup reduction");
      return nullptr;
    }
  }

  // According to the OpenCL Spec, we are allowed to rearrange the operation
  // order of a workgroup/subgroup reduction any way we like (even though
  // floating point addition is not associative so might not produce exactly
  // the same result), so we reduce to a single vector first, if necessary, and
  // then do a single reduction to scalar. This is more efficient than doing
  // multiple reductions to scalar and then BinOp'ing multiple scalars
  // together.
  //
  // Reduce to a single vector.
  while ((packetWidth >>= 1)) {
    for (decltype(packetWidth) i = 0; i < packetWidth; ++i) {
      Value *const lhs = opPackets[i];
      Value *const rhs = opPackets[i + packetWidth];
      opPackets[i] =
          multi_llvm::createBinOpForRecurKind(B, lhs, rhs, Info->Recurrence);
    }
  }

  // Reduce to a scalar.
  Value *v =
      createSimpleTargetReduction(B, &TTI, opPackets.front(), Info->Recurrence);

  // We leave the original reduction function and divert the vectorized
  // reduction through it, giving us a reduction over the full apparent
  // sub-group or work-group size (vecz * mux).
  CI->setOperand(argIdx, v);

  return CI;
}

Value *Packetizer::Impl::packetizeGroupBroadcast(Instruction *I) {
  auto *const CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->getCalledFunction()) {
    return nullptr;
  }
  compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  Function *callee = CI->getCalledFunction();
  auto const Builtin = BI.analyzeBuiltin(*callee);

  bool isWorkGroup = false;
  if (auto Info = BI.isMuxGroupCollective(Builtin.ID)) {
    if (!Info->isBroadcast() ||
        (!Info->isSubGroupScope() && !Info->isWorkGroupScope())) {
      return nullptr;
    }
    isWorkGroup = Info->isWorkGroupScope();
  } else {
    return nullptr;
  }

  IRBuilder<> B(CI);

  unsigned argIdx = isWorkGroup ? 1 : 0;
  auto *const src = CI->getArgOperand(argIdx);

  auto op = packetize(src);
  PACK_FAIL_IF(!op);

  // If the source operand happened to be a broadcast value already, we can use
  // it directly.
  if (op.info->numInstances == 0) {
    IC.deleteInstructionLater(CI);
    CI->replaceAllUsesWith(src);
    return src;
  }

  auto *idx = CI->getArgOperand(argIdx + 1);
  // We need to sanitize the input index so that it stays within the range of
  // one vectorized group.
  auto *const minVal =
      ConstantInt::get(idx->getType(), SimdWidth.getKnownMinValue());
  Value *idxFactor = minVal;
  if (SimdWidth.isScalable()) {
    idxFactor = B.CreateVScale(minVal);
  }
  idx = B.CreateURem(idx, idxFactor);

  Value *val = nullptr;
  // Optimize the constant fixed-vector case, where we can choose the exact
  // subpacket to extract from directly.
  if (isa<ConstantInt>(idx) && !SimdWidth.isScalable()) {
    ValuePacket opPackets;
    op.getPacketValues(opPackets);
    auto factor = SimdWidth.divideCoefficientBy(opPackets.size());
    const unsigned subvecSize = factor.getFixedValue();
    const unsigned idxVal = cast<ConstantInt>(idx)->getZExtValue();
    // If individual elements are scalar (through instantiation, say) then just
    // use the desired packet directly.
    if (subvecSize == 1) {
      val = opPackets[idxVal];
    } else {
      // Else extract from the correct packet, adjusting the index as we go.
      val = B.CreateExtractElement(
          opPackets[idxVal / subvecSize],
          ConstantInt::get(idx->getType(), idxVal % subvecSize));
    }
  } else {
    val = B.CreateExtractElement(op.getAsValue(), idx);
  }

  // We leave the origial broadcast function and divert the vectorized
  // broadcast through it, giving us a broadcast over the full apparent
  // sub-group or work-group size (vecz * mux).
  CI->setOperand(argIdx, val);

  return CI;
}

bool Packetizer::Impl::isSubgroupShuffle(Instruction *I) {
  auto *const CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->getCalledFunction()) {
    return false;
  }
  compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  Function *callee = CI->getCalledFunction();

  auto const Builtin = BI.analyzeBuiltin(*callee);
  auto const Info = BI.isMuxGroupCollective(Builtin.ID);

  return Info && Info->isSubGroupScope() && Info->isShuffleLike();
}

Value *Packetizer::Impl::packetizeMaskVarying(Instruction *I) {
  if (auto memop = MemOp::get(I)) {
    auto *const mask = memop->getMaskOperand();
    if (!mask) {
      return nullptr;
    }

    Value *vecMask = nullptr;

    MemOpDesc desc = memop->getDesc();
    bool isVector = desc.getDataType()->isVectorTy();

    // If only the mask operand is varying, we do not need to vectorize the
    // MemOp itself, only reduce the mask with an OR.
    if (!isVector) {
      vecMask = packetize(mask).getAsValue();
    } else {
      // If it's a vector, and the mask is splatted, then packetize the
      // splatted value, reduce it, then re-splat it as a vector. Otherwise, we
      // send it to the instantiator.
      auto *const splatVal = getSplatValue(mask);
      if (!splatVal) {
        return nullptr;
      }
      vecMask = packetize(splatVal).getAsValue();
    }

    VECZ_FAIL_IF(!vecMask);

    // Build the reduction right after the vector to reduce register
    // pressure, and to make it easier for CSE/GVN to combine them if there
    // are multiple uses of the same value (we could cache these?)
    auto *maskInst = dyn_cast<Instruction>(vecMask);
    IRBuilder<> B(maskInst ? buildAfter(maskInst, F) : I);

    // Sanitize any vector-predicated inputs.
    if (VL) {
      vecMask = sanitizeVPReductionInput(B, vecMask, VL, RecurKind::Or);
      VECZ_FAIL_IF(!vecMask);
    }

    Value *anyOfMask =
        createSimpleTargetReduction(B, &TTI, vecMask, RecurKind::Or);
    anyOfMask->setName("any_of_mask");

    if (isVector) {
      anyOfMask = B.CreateVectorSplat(
          multi_llvm::getVectorElementCount(desc.getDataType()), anyOfMask);
    }

    memop->setMaskOperand(anyOfMask);

    return I;
  }

  auto *const CI = dyn_cast<CallInst>(I);
  if (!CI) {
    return nullptr;
  }

  Function *callee = CI->getCalledFunction();

  // Handle internal builtins.
  if (Ctx.isInternalBuiltin(callee)) {
    // Handle lane mask reductions.
    // We treat these as Mask Varying instructions since their single argument
    // represents a lane mask and their result is a reduction over all lanes,
    // which means it is effectively uniform. We don't actually have to check
    // that they are mask varying, because that is the only possible uniformity
    // value of these function calls.
    compiler::utils::Lexer L(callee->getName());
    VECZ_FAIL_IF(!L.Consume(VectorizationContext::InternalBuiltinPrefix));
    bool any = false;
    bool divergence = false;
    if (L.Consume("divergence_any")) {
      divergence = true;
    } else if (L.Consume("divergence_all")) {
      any = true;
      divergence = true;
    }

    if (divergence) {
      IC.deleteInstructionLater(CI);
      auto *const reduce = reduceBranchCond(CI->getOperand(0), CI, any);
      CI->replaceAllUsesWith(reduce);
      return reduce;
    }
  }

  return nullptr;
}

ValuePacket Packetizer::Impl::packetizePHI(PHINode *Phi) {
  ValuePacket results;
  auto *const ty = Phi->getType();

  auto *wideTy = ty;
  unsigned packetWidth = 0;
  if (ty->isVectorTy() || VectorType::isValidElementType(ty)) {
    packetWidth = getPacketWidthForType(ty);
    wideTy =
        getWideType(Phi->getType(), SimdWidth.divideCoefficientBy(packetWidth));
  } else {
    // It's not a type we can widen, but we can save the instantiator the job..
    if (SimdWidth.isScalable()) {
      // as long as we aren't requesting a scalable vectorization factor..
      return results;
    }
    packetWidth = SimdWidth.getFixedValue();
  }

  IRBuilder<> B(buildAfter(Phi, F, true));
  auto numVals = Phi->getNumIncomingValues();
  auto name = Phi->getName();
  for (unsigned i = 0; i < packetWidth; ++i) {
    results.push_back(B.CreatePHI(wideTy, numVals, name));
  }

  // To avoid cycles in the use/def chain, packetize the incoming values later.
  // This allows packetizing phi uses by creating an 'empty' phi placeholder.
  pendingPhis.push_back(Phi);
  return results;
}

ValuePacket Packetizer::Impl::packetizeCall(CallInst *CI) {
  ValuePacket results;

  Function *Callee = CI->getCalledFunction();
  if (!Callee) {
    return results;
  }

  IRBuilder<> B(CI);
  // Handle LLVM intrinsics.
  if (Callee->isIntrinsic()) {
    auto IntrID = Intrinsic::ID(Callee->getIntrinsicID());
    if (IntrID == llvm::Intrinsic::lifetime_end ||
        IntrID == llvm::Intrinsic::lifetime_start) {
      auto *ptr = CI->getOperand(1);
      if (auto *const bcast = dyn_cast<BitCastInst>(ptr)) {
        ptr = bcast->getOperand(0);
      }

      if (auto *const alloca = dyn_cast<AllocaInst>(ptr)) {
        if (!needsInstantiation(Ctx, *alloca)) {
          // If it's an alloca we can widen, we can just change the size
          llvm::TypeSize const allocSize =
              Ctx.dataLayout()->getTypeAllocSize(alloca->getAllocatedType());
          auto const lifeSize = allocSize.isScalable() || SimdWidth.isScalable()
                                    ? -1
                                    : multi_llvm::getKnownMinValue(allocSize) *
                                          SimdWidth.getKnownMinValue();
          CI->setOperand(
              0, ConstantInt::get(CI->getOperand(0)->getType(), lifeSize));
          results.push_back(CI);
        }
      }
      return results;
    }

    auto const Props = Ctx.builtins().analyzeBuiltin(*Callee).properties;
    if (!(Props & compiler::utils::eBuiltinPropertyVectorEquivalent)) {
      return results;
    }

    // Only floating point intrinsics need this to be set to CI.
    // The IR Builder helpfully crashes when we pass it unnecessarily.
    Instruction *fastMathSrc = isa<FPMathOperator>(CI) ? CI : nullptr;

    // Using a native array with hard coded size for simplicity, make sure
    // to increase this if intrinsics with more operands are to be handled
    size_t constexpr maxOperands = 3;
    // Some llvm intrinsic functions like abs have argument that are constants
    // and define as llvm_i1_ty. This means that thoses operand can't
    // be packetized. To solve that temporary, we use this vector so every
    // cases can set independently what operand must be skipped
    // CA-3696
    SmallVector<bool, maxOperands> operandsToSkip(maxOperands, false);
    switch (IntrID) {
      case Intrinsic::abs:
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
        // def abs [LLVMMatchType<0>, llvm_i1_ty]
        operandsToSkip = {false, true};
        break;
      default:
        break;
    }

    auto *const ty = CI->getType();
    auto packetWidth = getPacketWidthForType(ty);
    auto *const wideTy =
        getWideType(ty, SimdWidth.divideCoefficientBy(packetWidth));

    auto const n = CI->arg_size();
    assert(n <= maxOperands && "Intrinsic has too many arguments");

    SmallVector<Value *, 16> opPackets[maxOperands];
    for (auto i = decltype(n){0}; i < n; ++i) {
      auto *argOperand = CI->getArgOperand(i);

      if (operandsToSkip[i]) {
        assert(isa<Constant>(argOperand) && "Operand should be a Constant");
        opPackets[i].resize(packetWidth);
        std::fill(opPackets[i].begin(), opPackets[i].end(), argOperand);
      } else {
        auto op = packetize(CI->getArgOperand(i));
        if (!op) {
          return results;
        }
        op.getPacketValues(packetWidth, opPackets[i]);
        PACK_FAIL_IF(opPackets[i].empty());
      }
    }

    auto const name = CI->getName();
    Type *const types[1] = {wideTy};  // because LLVM 13 is a numpty
    Value *opVals[maxOperands];
    for (unsigned i = 0; i < packetWidth; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        opVals[j] = opPackets[j][i];
      }

      results.push_back(B.CreateIntrinsic(
          IntrID, types, ArrayRef<Value *>(opVals, n), fastMathSrc, name));
    }
    return results;
  }

  // Handle internal builtins.
  if (Ctx.isInternalBuiltin(Callee)) {
    // Handle masked loads and stores.
    if (auto MaskedOp = MemOp::get(CI, MemOpAccessKind::Masked)) {
      if (MaskedOp->isMaskedMemOp()) {
        return packetizeMemOp(*MaskedOp);
      }
    }
  }

  auto const Builtin = Ctx.builtins().analyzeBuiltin(*Callee);

  // Handle subgroup scans, which defer to internal builtins.
  if (auto Info = Ctx.builtins().isMuxGroupCollective(Builtin.ID)) {
    if (Info->isSubGroupScope() && Info->isScan()) {
      return packetizeSubgroupScan(CI, *Info);
    }
  }

  // Handle external builtins.
  auto const Props = Builtin.properties;
  if (Props & compiler::utils::eBuiltinPropertyExecutionFlow ||
      Props & compiler::utils::eBuiltinPropertyWorkItem) {
    return results;
  }

  auto *const ty = CI->getType();

  // Our builtins are only defined up to a width of 16 so will not vectorize
  // above that. Inspect the operands as well in case they are wider, for
  // instance a convert from float to i8, we would rather widen according to
  // the float and not the i8 so we don't create too wide a vector of floats.
  auto packetWidth = getPacketWidthForType(ty, 16u);
  for (const auto &op : CI->data_ops()) {
    auto *const vTy = op.get()->getType();
    if (!vTy->isPointerTy()) {
      packetWidth = std::max(packetWidth, getPacketWidthForType(vTy, 16u));
    }
  }

  auto factor = SimdWidth.divideCoefficientBy(packetWidth);

  // Try to find a unit for this builtin.
  auto CalleeVec = Ctx.getVectorizedFunction(*Callee, factor);
  if (!CalleeVec) {
    // No vectorization strategy found. Fall back on Instantiation.
    return results;
  }

  // Packetize call operands.
  // But not if they have pointer return arguments (handled in vectorizeCall).
  for (const auto &TargetArg : CalleeVec.args) {
    PACK_FAIL_IF(TargetArg.kind == VectorizationResult::Arg::POINTER_RETURN);
  }

  auto *const vecTy = dyn_cast<FixedVectorType>(ty);
  unsigned const scalarWidth = vecTy ? vecTy->getNumElements() : 1;
  unsigned i = 0;
  SmallVector<SmallVector<Value *, 16>, 4> opPackets;
  for (const auto &TargetArg : CalleeVec.args) {
    opPackets.emplace_back();

    // Handle scalar arguments.
    Value *scalarOp = CI->getArgOperand(i);
    if (TargetArg.kind == VectorizationResult::Arg::SCALAR) {
      for (unsigned j = 0; j < packetWidth; ++j) {
        opPackets.back().push_back(scalarOp);
      }
      i++;
      continue;
    }

    // Vectorize scalar operands.
    auto op = packetize(CI->getOperand(i));
    PACK_FAIL_IF(!op);

    // The vector versions of some builtins can have a mix of vector and scalar
    // arguments. We need to widen any scalar arguments by sub-splatting.
    auto *const scalarTy = scalarOp->getType();
    auto *const argTy = TargetArg.type;
    if (vecTy && !scalarTy->isVectorTy()) {
      PACK_FAIL_IF(argTy->getScalarType() != scalarTy);

      op.getPacketValues(packetWidth, opPackets.back());
      PACK_FAIL_IF(opPackets.back().empty());

      // Widen the scalar operands.
      PACK_FAIL_IF(
          !createSubSplats(Ctx.targetInfo(), B, opPackets.back(), scalarWidth));
    } else {
      // Make sure the type is correct for vector arguments.
      Type *wideTy = getWideType(scalarOp->getType(), factor);
      PACK_FAIL_IF(argTy != wideTy);

      op.getPacketValues(packetWidth, opPackets.back());
      PACK_FAIL_IF(opPackets.back().empty());
    }
    i++;
  }

  auto numArgs = opPackets.size();
  SmallVector<Value *, 4> opVals;
  opVals.resize(numArgs);

  auto *vecFn = CalleeVec.get();
  for (unsigned i = 0; i < packetWidth; ++i) {
    for (unsigned j = 0; j < numArgs; ++j) {
      opVals[j] = opPackets[j][i];
    }

    CallInst *newCI = B.CreateCall(vecFn, opVals, CI->getName());
    newCI->setCallingConv(CI->getCallingConv());
    results.push_back(newCI);
  }

  return results;
}

ValuePacket Packetizer::Impl::packetizeSubgroupScan(
    CallInst *CI, compiler::utils::GroupCollective Scan) {
  ValuePacket results;

  Function *callee = CI->getCalledFunction();
  if (!callee) {
    return results;
  }

  compiler::utils::NameMangler mangler(&CI->getContext());

  // The operands and types for the internal builtin
  SmallVector<Value *, 2> Ops = {packetize(CI->getArgOperand(0)).getAsValue()};
  SmallVector<Type *, 2> Tys = {getWideType(CI->getType(), SimdWidth)};

  bool isInclusive =
      Scan.Op == compiler::utils::GroupCollective::OpKind::ScanInclusive;
  StringRef op = "add";
  // min/max scans are prefixed with s/u if they are signed/unsigned integer
  // operations. The value 'None' here represents an operation where the sign
  // of the operands is unimportant, such as floating-point operations, or
  // integer addition.
  bool opIsSignedInt = false;

  switch (Scan.Recurrence) {
    default:
      assert(false && "Impossible subgroup scan kind");
      return results;
    case llvm::RecurKind::Add:
    case llvm::RecurKind::FAdd:
      op = "add";
      break;
    case llvm::RecurKind::SMin:
      op = "smin";
      opIsSignedInt = true;
      break;
    case llvm::RecurKind::UMin:
      op = "umin";
      break;
    case llvm::RecurKind::FMin:
      op = "min";
      break;
    case llvm::RecurKind::SMax:
      op = "smax";
      opIsSignedInt = true;
      break;
    case llvm::RecurKind::UMax:
      op = "umax";
      break;
    case llvm::RecurKind::FMax:
      op = "max";
      break;
    case llvm::RecurKind::Mul:
    case llvm::RecurKind::FMul:
      op = "mul";
      break;
    case llvm::RecurKind::And:
      op = "and";
      break;
    case llvm::RecurKind::Or:
      op = "or";
      break;
    case llvm::RecurKind::Xor:
      op = "xor";
      break;
  }

  // Now create the mangled builtin function name.
  SmallString<128> NameSV;
  raw_svector_ostream O(NameSV);

  // We don't bother with VP for fixed vectors, because it doesn't save us
  // anything.
  bool const VP = VL && SimdWidth.isScalable();

  O << VectorizationContext::InternalBuiltinPrefix << "sub_group_scan_"
    << (isInclusive ? "inclusive" : "exclusive") << "_" << op
    << (VP ? "_vp" : "") << "_";

  compiler::utils::TypeQualifiers VecQuals(
      compiler::utils::eTypeQualNone, opIsSignedInt
                                          ? compiler::utils::eTypeQualSignedInt
                                          : compiler::utils::eTypeQualNone);
  if (!mangler.mangleType(O, Tys[0], VecQuals)) {
    return results;
  }

  // VP operations mangle the extra i32 VL operand.
  if (VP) {
    Ops.push_back(VL);
    Tys.push_back(VL->getType());
    compiler::utils::TypeQualifiers VLQuals(compiler::utils::eTypeQualNone);
    if (!mangler.mangleType(O, Tys[1], VLQuals)) {
      return results;
    }
  }

  auto *SubgroupScanFnTy = FunctionType::get(Tys[0], Tys, /*isVarArg*/ false);
  auto *const SubgroupFn =
      Ctx.getOrCreateInternalBuiltin(NameSV, SubgroupScanFnTy);

  IRBuilder<> B(CI);

  auto *VectorScan = B.CreateCall(SubgroupFn, Ops);

  // We've currently got a scan over each vector group, but the full sub-group
  // is further multiplied by the mux sub-group size. For example, we may have
  // a vectorization factor sized group of 4 and a mux sub-group size of 2.
  // Together the full sub-group size to the user is 4*2 = 8.
  // In terms of invocations, we've essentially currently got:
  //   <a0, a0+a1, a0+a1+a2, a0+a1+a2+a3> (invocation 0)
  //   <a4, a4+a5, a4+a5+a6, a4+a5+a6+a7> (invocation 1)
  // These two iterations need to be further scanned over the mux sub-group
  // size. We do this by adding the identity to the first invocation, the
  // result of the scan over the first invocation to the second, etc. This is
  // an exclusive scan over the *reduction* of the input vector:
  //   <a0, a1, a2, a3> (invocation 0)
  //   <a4, a5, a6, a7> (invocation 1)
  // -> reduction
  //   (a0+a1+a2+a3) (invocation 0)
  //   (a4+a5+a6+a7) (invocation 1)
  // -> exclusive mux sub-group scan
  //               I (invocation 0)
  //   (a0+a1+a2+a3) (invocation 1)
  // -> adding that to the result of the vector scan:
  //   <I+a0, I+a0+a1, I+a0+a1+a2, I+a0+a1+a2+a3>          (invocation 0)
  //   <(a0+a1+a2+a3)+a4, (a0+a1+a2+a3)+a4+a5,             (invocation 1)
  //    (a0+a1+a2+a3)+a4+a5+a6, (a0+a1+a2+a3)+a4+a5+a6+a7>
  // When viewed as a full 8-element vector, this is our final scan.
  // Thus we essentially keep the original mux sub-group scan, but change it to
  // be an exclusive one.
  auto *Reduction = Ops.front();
  if (VL) {
    Reduction = sanitizeVPReductionInput(B, Reduction, VL, Scan.Recurrence);
    if (!Reduction) {
      return results;
    }
  }
  Reduction = createSimpleTargetReduction(B, &TTI, Reduction, Scan.Recurrence);

  // Now we defer to an *exclusive* scan over the mux sub-group.
  auto ExclScan = Scan;
  ExclScan.Op = compiler::utils::GroupCollective::OpKind::ScanExclusive;

  auto ExclScanID = Ctx.builtins().getMuxGroupCollective(ExclScan);
  assert(ExclScanID != compiler::utils::eBuiltinInvalid);

  auto *const ExclScanFn = Ctx.builtins().getOrDeclareMuxBuiltin(
      ExclScanID, *F.getParent(), {CI->getType()});
  assert(ExclScanFn);

  auto *const ExclScanCI = B.CreateCall(ExclScanFn, {Reduction});

  Value *const Splat = B.CreateVectorSplat(SimdWidth, ExclScanCI);

  auto *const Result = multi_llvm::createBinOpForRecurKind(B, VectorScan, Splat,
                                                           Scan.Recurrence);

  results.push_back(Result);
  return results;
}

Value *Packetizer::Impl::vectorizeInstruction(Instruction *Ins) {
  if (needsInstantiation(Ctx, *Ins)) {
    return nullptr;
  }

  // Figure out what kind of instruction it is and try to vectorize it.
  Value *Result = nullptr;
  switch (Ins->getOpcode()) {
    default:
      break;
    case Instruction::Call:
      Result = vectorizeCall(cast<CallInst>(Ins));
      break;
    case Instruction::Ret:
      Result = vectorizeReturn(cast<ReturnInst>(Ins));
      break;
    case Instruction::Alloca:
      Result = vectorizeAlloca(cast<AllocaInst>(Ins));
      break;
    case Instruction::ExtractValue:
      Result = vectorizeExtractValue(cast<ExtractValueInst>(Ins));
      break;
  }

  if (Result) {
    vectorizeDI(Ins, Result);
  }
  return Result;
}

ValuePacket Packetizer::Impl::packetizeLoad(LoadInst *Load) {
  auto Op = *MemOp::get(Load);
  return packetizeMemOp(Op);
}

ValuePacket Packetizer::Impl::packetizeStore(StoreInst *Store) {
  auto Op = *MemOp::get(Store);
  return packetizeMemOp(Op);
}

ValuePacket Packetizer::Impl::packetizeMemOp(MemOp &op) {
  ValuePacket results;

  // Determine the stride of the memory operation.
  // Vectorize the pointer if there is no valid stride.
  Value *ptr = op.getPointerOperand();
  assert(ptr && "Could not get pointer operand of Op");

  auto *const dataTy = op.getDataType();
  if (!dataTy->isVectorTy() && !VectorType::isValidElementType(dataTy)) {
    return results;
  }

  if (auto *const vecTy = dyn_cast<FixedVectorType>(dataTy)) {
    auto const elts = vecTy->getNumElements();
    if (elts & (elts - 1)) {
      // If the data type is a vector with number of elements not a power of 2,
      // it is not safe to widen, because of alignment padding. Reject it and
      // let instantiation deal with it..
      return results;
    }
  }

  auto const packetWidth = getPacketWidthForType(dataTy);
  // Note: NOT const because LLVM 11 can't multiply a const ElementCount.
  auto factor = SimdWidth.divideCoefficientBy(packetWidth);

  if (factor.isScalar()) {
    // not actually widening anything here, so just instantiate it
    return results;
  }

  if (VL && packetWidth != 1) {
    emitVeczRemarkMissed(&F, op.getInstr(),
                         "Can not vector-predicate packets larger than 1");
    return {};
  }

  IRBuilder<> B(op.getInstr());
  IC.deleteInstructionLater(op.getInstr());

  auto const name = op.getInstr()->getName();
  auto *const mask = op.getMaskOperand();
  auto *const data = op.getDataOperand();
  auto *const stride = SAR.buildMemoryStride(B, ptr, dataTy);

  auto *const vecPtrTy = dyn_cast<FixedVectorType>(dataTy);

  // If we're vector-predicating a vector access, scale the vector length up by
  // the original number of vector elements.
  // Adjust the MemOp so that it is VL-predicated, if we must.
  Value *EVL = VL;
  if (vecPtrTy && VL) {
    EVL = B.CreateMul(VL, B.getInt32(vecPtrTy->getNumElements()));
  }

  auto *const constantStrideVal = dyn_cast_or_null<ConstantInt>(stride);
  int constantStride =
      constantStrideVal ? constantStrideVal->getSExtValue() : 0;
  bool validStride = stride && (!constantStrideVal || constantStride != 0);
  if (!validStride) {
    if (dataTy->isPointerTy()) {
      // We do not have vector-of-pointers support in Vecz builtins, hence
      // instantiate instead of packetize
      return results;
    }

    bool const scalable = SimdWidth.isScalable();
    if (!mask && dataTy->isVectorTy() && !scalable) {
      // unmasked scatter/gathers are better off instantiated..
      return results;
    }

    // Assume that individual masked loads/stores are more efficient when the
    // type does not fit into a native integer. Since instantiation is never an
    // option for scalable vectors, we do not consider this option.
    if (vecPtrTy && !scalable &&
        !Ctx.dataLayout()->fitsInLegalInteger(
            dataTy->getPrimitiveSizeInBits())) {
      return results;
    }

    auto ptrPacket = packetizeAndGet(ptr, packetWidth);
    PACK_FAIL_IF(ptrPacket.empty());

    auto *const scalarTy = dataTy->getScalarType();
    auto *const scalarPtrTy =
        cast<PointerType>(ptr->getType()->getScalarType());

    // When scattering/gathering with a vector type, we can cast it to a
    // vector of pointers to the scalar type and widen it into a vector
    // of pointers to all the individual elements, and then gather/scatter
    // using that.
    if (vecPtrTy && scalable) {
      // Scalable requires special codegen that avoids shuffles, but the idea
      // is the same.
      // We only handle the one packet right now.
      PACK_FAIL_IF(ptrPacket.size() != 1);
      auto const scalarWidth = vecPtrTy->getNumElements();
      Value *&vecPtr = ptrPacket.front();
      ElementCount const wideEC = factor * scalarWidth;
      // Sub-splat the pointers such that we get, e.g.:
      // <A, B> -> x4 -> <A, A, A, A, B, B, B, B>
      bool const success =
          createSubSplats(Ctx.targetInfo(), B, ptrPacket, scalarWidth);
      PACK_FAIL_IF(!success);
      auto *const newPtrTy = llvm::VectorType::get(
          PointerType::get(scalarTy, scalarPtrTy->getPointerAddressSpace()),
          wideEC);
      // Bitcast the above sub-splat to purely scalar pointers
      vecPtr = B.CreateBitCast(vecPtr, newPtrTy);
      // Create an index sequence to start the offseting process
      Value *idxVector = multi_llvm::createIndexSequence(
          B, VectorType::get(B.getInt32Ty(), wideEC), wideEC, "index.vec");
      PACK_FAIL_IF(!idxVector);
      // Modulo the indices 0,1,2,.. with the original vector type, producing,
      // e.g., for the above: <0,1,2,3,0,1,2,3>
      auto *const subVecEltsSplat =
          B.CreateVectorSplat(wideEC, B.getInt32(scalarWidth));
      idxVector = B.CreateURem(idxVector, subVecEltsSplat);
      // Index into the pointer vector with the offsets, e.g.,:
      // <A, A+1, A+2, A+3, B, B+1, B+2, B+3>
      vecPtr = B.CreateInBoundsGEP(scalarTy, vecPtr, idxVector);
    } else if (vecPtrTy && !scalable) {
      auto const simdWidth = factor.getFixedValue();
      auto const scalarWidth = vecPtrTy->getNumElements();

      // Build shuffle mask to widen the pointer
      SmallVector<Constant *, 16> indices;
      SmallVector<int, 16> widenMask;
      for (size_t i = 0; i < simdWidth; ++i) {
        for (size_t j = 0; j < scalarWidth; ++j) {
          widenMask.push_back(i);
          indices.push_back(B.getInt32(j));
        }
      }

      auto *const newPtrTy = FixedVectorType::get(
          PointerType::get(scalarTy, scalarPtrTy->getPointerAddressSpace()),
          simdWidth);

      auto *const idxVector = ConstantVector::get(indices);
      auto *const undef = UndefValue::get(newPtrTy);
      for (auto &vecPtr : ptrPacket) {
        vecPtr = B.CreateBitCast(vecPtr, newPtrTy);
        vecPtr = B.CreateShuffleVector(vecPtr, undef, widenMask);
        vecPtr = B.CreateInBoundsGEP(scalarTy, vecPtr, idxVector);
      }
    }

    ValuePacket dataPacket;
    if (data) {
      auto src = packetize(data);
      PACK_FAIL_IF(!src);
      src.getPacketValues(packetWidth, dataPacket);
      PACK_FAIL_IF(dataPacket.empty());
    } else {
      dataPacket.resize(packetWidth, nullptr);
    }

    // Vector-predicated scatters/gathers are always masked.
    ValuePacket maskPacket(packetWidth, nullptr);
    auto *const packetVecTy = getWideType(dataTy, factor);
    if (mask || EVL) {
      if (!mask) {
        // If there's no mask then just splat a trivial one.
        auto *const trueMask = multi_llvm::createAllTrueMask(
            B, multi_llvm::getVectorElementCount(packetVecTy));
        std::fill(maskPacket.begin(), maskPacket.end(), trueMask);
      } else {
        maskPacket = packetizeAndGet(mask, packetWidth);
        PACK_FAIL_IF(maskPacket.empty());
      }
    }

    // Gather load or scatter store.
    for (unsigned i = 0; i != packetWidth; ++i) {
      if (op.isLoad()) {
        results.push_back(createGather(Ctx, packetVecTy, ptrPacket[i],
                                       maskPacket[i], EVL, op.getAlignment(),
                                       name, op.getInstr()));
      } else {
        results.push_back(createScatter(Ctx, dataPacket[i], ptrPacket[i],
                                        maskPacket[i], EVL, op.getAlignment(),
                                        name, op.getInstr()));
      }
    }
  } else if (!constantStrideVal || constantStride != 1) {
    if (dataTy->isPointerTy() || dataTy->isVectorTy()) {
      // No builtins for memops on pointer types, and we can't do interleaved
      // memops over vector types.
      return results;
    }

    ValuePacket dataPacket;
    if (data) {
      auto src = packetize(data);
      PACK_FAIL_IF(!src);
      src.getPacketValues(packetWidth, dataPacket);
      PACK_FAIL_IF(dataPacket.empty());
    } else {
      dataPacket.resize(packetWidth, nullptr);
    }

    Value *packetStride = nullptr;
    if (packetWidth != 1) {
      // Make sure the stride is at least as wide as a GEP index needs to be
      unsigned const indexBits = Ctx.dataLayout()->getIndexSizeInBits(
          ptr->getType()->getPointerAddressSpace());
      unsigned strideBits = stride->getType()->getPrimitiveSizeInBits();
      auto *const elementStride =
          (indexBits > strideBits)
              ? B.CreateSExt(stride, B.getIntNTy((strideBits = indexBits)))
              : stride;

      auto const simdWidth = factor.getFixedValue();
      packetStride =
          B.CreateMul(elementStride, B.getIntN(strideBits, simdWidth),
                      Twine(name, ".packet_stride"));
    }

    // Vector-predicated interleaved operations are always masked.
    ValuePacket maskPacket(packetWidth, nullptr);
    auto *const packetVecTy = getWideType(dataTy, factor);
    if (mask || EVL) {
      if (!mask) {
        // If there's no mask then just splat a trivial one.
        auto *const trueMask = multi_llvm::createAllTrueMask(
            B, multi_llvm::getVectorElementCount(packetVecTy));
        std::fill(maskPacket.begin(), maskPacket.end(), trueMask);
      } else {
        maskPacket = packetizeAndGet(mask, packetWidth);
        PACK_FAIL_IF(maskPacket.empty());
      }
    }

    // Interleaved (strided) load or store.
    for (unsigned i = 0; i != packetWidth; ++i) {
      if (i != 0) {
        ptr = B.CreateInBoundsGEP(dataTy, ptr, packetStride,
                                  Twine(name, ".incr"));
      }
      if (op.isLoad()) {
        results.push_back(
            createInterleavedLoad(Ctx, packetVecTy, ptr, stride, maskPacket[i],
                                  EVL, op.getAlignment(), name, op.getInstr()));
      } else {
        results.push_back(createInterleavedStore(
            Ctx, dataPacket[i], ptr, stride, maskPacket[i], EVL,
            op.getAlignment(), name, op.getInstr()));
      }
    }
  } else {
    ValuePacket dataPacket;
    if (data) {
      auto src = packetize(data);
      PACK_FAIL_IF(!src);
      src.getPacketValues(packetWidth, dataPacket);
      PACK_FAIL_IF(dataPacket.empty());
    } else if (mask) {
      // don't need the data packet for unmasked stores
      dataPacket.resize(packetWidth, nullptr);
    }

    Value *packetStride = nullptr;
    if (packetWidth != 1) {
      auto const simdWidth = factor.getFixedValue();
      packetStride = B.getInt64(simdWidth);
    }

    // Calculate the alignment. The MemOp's alignment is the original
    // alignment, but may be overaligned. After vectorization it can't be
    // larger than the pointee element type.
    unsigned alignment = op.getAlignment();
    unsigned sizeInBits =
        multi_llvm::getKnownMinValue(dataTy->getPrimitiveSizeInBits());
    alignment = std::min(alignment, std::max(sizeInBits, 8u) / 8u);

    // Regular load or store.
    if (mask) {
      bool isVectorMask = mask->getType()->isVectorTy();
      auto maskPacket = packetizeAndGet(mask, packetWidth);
      PACK_FAIL_IF(maskPacket.empty());

      // If the original instruction was a vector but the mask was a scalar i1,
      // we have to broadcast the mask elements across the data vector.
      auto *const vecTy = dyn_cast<FixedVectorType>(dataTy);
      if (vecTy && !isVectorMask) {
        PACK_FAIL_IF(factor.isScalable());
        unsigned simdWidth = factor.getFixedValue();
        unsigned scalarWidth = vecTy->getNumElements();

        // Build shuffle mask to widen the vector condition.
        SmallVector<int, 16> widenMask;
        for (size_t i = 0; i < simdWidth; ++i) {
          for (size_t j = 0; j < scalarWidth; ++j) {
            widenMask.push_back(i);
          }
        }

        auto *const undef = UndefValue::get(maskPacket.front()->getType());
        for (auto &vecMask : maskPacket) {
          vecMask = createOptimalShuffle(B, vecMask, undef, widenMask);
        }
      }

      for (unsigned i = 0; i != packetWidth; ++i) {
        if (i != 0) {
          ptr = B.CreateInBoundsGEP(dataTy, ptr, packetStride,
                                    Twine(name, ".incr"));
        }
        if (op.isLoad()) {
          results.push_back(createMaskedLoad(
              Ctx, getWideType(dataTy, factor), ptr, maskPacket[i], EVL,
              op.getAlignment(), name, op.getInstr()));
        } else {
          results.push_back(
              createMaskedStore(Ctx, dataPacket[i], ptr, maskPacket[i], EVL,
                                op.getAlignment(), name, op.getInstr()));
        }
      }
    } else {
      TargetInfo &VTI = Ctx.targetInfo();
      if (op.isLoad()) {
        auto *const one = B.getInt64(1);
        for (unsigned i = 0; i != packetWidth; ++i) {
          if (i != 0) {
            ptr = B.CreateInBoundsGEP(dataTy, ptr, packetStride,
                                      Twine(name, ".incr"));
          }
          results.push_back(
              VTI.createLoad(B, getWideType(dataTy, factor), ptr, one, EVL));
        }
      } else {
        auto *const one = B.getInt64(1);
        for (unsigned i = 0; i != packetWidth; ++i) {
          if (i != 0) {
            ptr = B.CreateInBoundsGEP(dataTy, ptr, packetStride,
                                      Twine(name, ".incr"));
          }
          results.push_back(
              VTI.createStore(B, dataPacket[i], ptr, one, alignment, EVL));
        }
      }
    }
  }

  // Transfer attributes from an old call instruction to a new one.
  if (CallInst *oldCI = op.getCall()) {
    for (auto *r : results) {
      if (CallInst *newCI = dyn_cast_or_null<CallInst>(r)) {
        newCI->setCallingConv(oldCI->getCallingConv());
      }
    }
  }
  return results;
}

void Packetizer::Impl::vectorizeDI(Instruction *, Value *) {
  // FIXME: Reinstate support for vectorizing debug info
  return;
}

ValuePacket Packetizer::Impl::packetizeGEP(GetElementPtrInst *GEP) {
  ValuePacket results;
  Value *pointer = GEP->getPointerOperand();
  if (isa<AllocaInst>(pointer)) {
    return results;
  }

  if (isa<VectorType>(GEP->getType())) {
    // instantiate vector GEPs, for safety
    return results;
  }

  // Work out the packet width from the pointed to type, rather than the
  // pointer type itself, because this is the width the memops will be using.
  auto *const ty = GEP->getSourceElementType();
  auto const packetWidth = getPacketWidthForType(ty);

  // It is legal to create a GEP with a mixture of scalar and vector operands.
  // If any operand is a vector, the result will be a vector of pointers.
  ValuePacket pointerPacket;
  if (UVR.isVarying(pointer)) {
    auto res = packetize(pointer);
    PACK_FAIL_IF(!res);
    res.getPacketValues(packetWidth, pointerPacket);
    PACK_FAIL_IF(pointerPacket.empty());
  } else {
    for (unsigned i = 0; i != packetWidth; ++i) {
      pointerPacket.push_back(pointer);
    }
  }

  // Packetize the GEP indices.
  SmallVector<SmallVector<Value *, 16>, 4> opPackets;
  for (unsigned i = 0, n = GEP->getNumIndices(); i != n; i++) {
    Value *idx = GEP->getOperand(i + 1);
    opPackets.emplace_back();

    // Handle constant indices
    if (isa<ConstantInt>(idx)) {
      for (unsigned j = 0; j < packetWidth; ++j) {
        opPackets.back().push_back(idx);
      }
    } else {
      auto op = packetize(idx);
      PACK_FAIL_IF(!op);
      op.getPacketValues(packetWidth, opPackets.back());
      PACK_FAIL_IF(opPackets.back().empty());
    }
  }

  IRBuilder<> B(GEP);
  IC.deleteInstructionLater(GEP);

  bool inBounds = GEP->isInBounds();
  auto const name = GEP->getName();

  auto const numIndices = opPackets.size();
  SmallVector<Value *, 4> opVals;
  opVals.resize(numIndices);
  for (unsigned i = 0; i < packetWidth; ++i) {
    for (unsigned j = 0; j < numIndices; ++j) {
      opVals[j] = opPackets[j][i];
    }

    if (inBounds) {
      results.push_back(
          B.CreateInBoundsGEP(ty, pointerPacket[i], opVals, name));
    } else {
      results.push_back(B.CreateGEP(ty, pointerPacket[i], opVals, name));
    }
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeBinaryOp(BinaryOperator *BinOp) {
  ValuePacket results;
  auto packetWidth = getPacketWidthForType(BinOp->getType());

  auto LHS = packetizeAndGet(BinOp->getOperand(0), packetWidth);
  auto RHS = packetizeAndGet(BinOp->getOperand(1), packetWidth);
  PACK_FAIL_IF(LHS.empty() || RHS.empty());

  auto opcode = BinOp->getOpcode();
  auto name = BinOp->getName();
  IRBuilder<> B(BinOp);
  if (VL) {
    auto *const VecTy = LHS[0]->getType();
    // Support for VP legalization is still lacking so fall back to non-VP
    // operations in other cases. This support will improve over time.
    if (Ctx.targetInfo().isVPVectorLegal(F, VecTy)) {
      PACK_FAIL_IF(packetWidth != 1);
      auto VPId = VPIntrinsic::getForOpcode(opcode);
      PACK_FAIL_IF(VPId == Intrinsic::not_intrinsic);
      auto *const Mask = multi_llvm::createAllTrueMask(
          B, multi_llvm::getVectorElementCount(LHS[0]->getType()));
      // Scale the base length by the number of vector elements, where
      // appropriate.
      Value *EVL = VL;
      if (auto *const VecTy = dyn_cast<VectorType>(BinOp->getType())) {
        EVL = B.CreateMul(
            EVL,
            B.getInt32(
                multi_llvm::getVectorElementCount(VecTy).getKnownMinValue()));
      }
      auto *const NewBinOp = B.CreateIntrinsic(VPId, {LHS[0]->getType()},
                                               {LHS[0], RHS[0], Mask, EVL});
      NewBinOp->copyIRFlags(BinOp, true);
      NewBinOp->copyMetadata(*BinOp);
      results.push_back(NewBinOp);
      return results;
    }
    // If we haven't matched [us]div or [us]rem then we may be executing
    // out-of-bounds elements if we don't predicate. Since this isn't safe,
    // bail.
    PACK_FAIL_IF(
        opcode == BinaryOperator::UDiv || opcode == BinaryOperator::SDiv ||
        opcode == BinaryOperator::URem || opcode == BinaryOperator::SRem);
  }
  for (unsigned i = 0; i < packetWidth; ++i) {
    auto *const NewV = B.CreateBinOp(opcode, LHS[i], RHS[i], name);
    if (auto *const NewBinOp = dyn_cast<BinaryOperator>(NewV)) {
      NewBinOp->copyIRFlags(BinOp, true);
      NewBinOp->copyMetadata(*BinOp);
    }
    results.push_back(NewV);
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeFreeze(FreezeInst *FreezeI) {
  ValuePacket results;
  auto resC = packetize(FreezeI->getOperand(0));
  PACK_FAIL_IF(!resC);

  SmallVector<Value *, 16> src;
  resC.getPacketValues(src);
  PACK_FAIL_IF(src.empty());

  auto const packetWidth = src.size();
  auto const name = FreezeI->getName();

  IRBuilder<> B(FreezeI);
  for (unsigned i = 0; i < packetWidth; ++i) {
    results.push_back(B.CreateFreeze(src[i], name));
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeUnaryOp(UnaryOperator *UnOp) {
  ValuePacket results;

  auto opcode = UnOp->getOpcode();

  auto packetWidth = getPacketWidthForType(UnOp->getType());
  auto src = packetizeAndGet(UnOp->getOperand(0), packetWidth);
  PACK_FAIL_IF(src.empty());

  auto name = UnOp->getName();
  IRBuilder<> B(UnOp);
  for (unsigned i = 0; i < packetWidth; ++i) {
    Value *New = B.CreateUnOp(opcode, src[i], name);
    auto *NewUnOp = cast<UnaryOperator>(New);
    NewUnOp->copyIRFlags(UnOp, true);
    results.push_back(NewUnOp);
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeCast(CastInst *CastI) {
  ValuePacket results;

  auto *const ty = CastI->getType();
  auto packetWidth = std::max(getPacketWidthForType(ty),
                              getPacketWidthForType(CastI->getSrcTy()));

  auto src = packetizeAndGet(CastI->getOperand(0), packetWidth);
  PACK_FAIL_IF(src.empty());

  auto *const wideTy =
      getWideType(ty, SimdWidth.divideCoefficientBy(packetWidth));
  auto name = CastI->getName();
  IRBuilder<> B(CastI);
  for (unsigned i = 0; i < packetWidth; ++i) {
    results.push_back(B.CreateCast(CastI->getOpcode(), src[i], wideTy, name));
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeICmp(ICmpInst *Cmp) {
  ValuePacket results;
  auto packetWidth = getPacketWidthForType(Cmp->getOperand(0)->getType());

  auto LHS = packetizeAndGet(Cmp->getOperand(0), packetWidth);
  auto RHS = packetizeAndGet(Cmp->getOperand(1), packetWidth);
  PACK_FAIL_IF(LHS.empty() || RHS.empty());

  auto pred = Cmp->getPredicate();
  auto name = Cmp->getName();
  IRBuilder<> B(Cmp);
  for (unsigned i = 0; i < packetWidth; ++i) {
    auto *const NewICmp = B.CreateICmp(pred, LHS[i], RHS[i], name);
    if (isa<ICmpInst>(NewICmp)) {
      cast<ICmpInst>(NewICmp)->copyIRFlags(Cmp, true);
    }
    results.push_back(NewICmp);
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeFCmp(FCmpInst *Cmp) {
  ValuePacket results;
  auto packetWidth = getPacketWidthForType(Cmp->getOperand(0)->getType());

  auto LHS = packetizeAndGet(Cmp->getOperand(0), packetWidth);
  auto RHS = packetizeAndGet(Cmp->getOperand(1), packetWidth);
  PACK_FAIL_IF(LHS.empty() || RHS.empty());

  auto pred = Cmp->getPredicate();
  auto name = Cmp->getName();
  IRBuilder<> B(Cmp);
  for (unsigned i = 0; i < packetWidth; ++i) {
    auto *NewICmp = cast<FCmpInst>(B.CreateFCmp(pred, LHS[i], RHS[i], name));
    NewICmp->copyIRFlags(Cmp, true);
    results.push_back(NewICmp);
  }
  return results;
}

ValuePacket Packetizer::Impl::packetizeSelect(SelectInst *Select) {
  ValuePacket results;
  auto *const ty = Select->getType();
  if (!ty->isVectorTy() && !VectorType::isValidElementType(ty)) {
    // Selects can work on struct/aggregate types, but we can't widen them..
    return results;
  }

  auto packetWidth = getPacketWidthForType(ty);
  auto vecT = packetizeAndGet(Select->getOperand(1), packetWidth);
  auto vecF = packetizeAndGet(Select->getOperand(2), packetWidth);
  PACK_FAIL_IF(vecT.empty() || vecF.empty());

  auto *cond = Select->getOperand(0);
  auto resC = packetize(cond);
  PACK_FAIL_IF(!resC);

  IRBuilder<> B(Select);
  bool isVectorSelect = cond->getType()->isVectorTy();
  SmallVector<Value *, 16> vecC;
  if (UVR.isVarying(cond)) {
    resC.getPacketValues(packetWidth, vecC);
    PACK_FAIL_IF(vecC.empty());

    // If the original select returns a vector, but the condition was scalar,
    // and its packet members are widened, we have to sub-broadcast it across
    // the lanes of the original vector.
    if (!isVectorSelect && vecC.front()->getType()->isVectorTy()) {
      if (auto *vecTy = dyn_cast<FixedVectorType>(Select->getType())) {
        PACK_FAIL_IF(!createSubSplats(Ctx.targetInfo(), B, vecC,
                                      vecTy->getNumElements()));
      }
    }
  } else if (isVectorSelect) {
    // If the condition is a uniform vector, get its broadcast packets
    resC.getPacketValues(packetWidth, vecC);
    PACK_FAIL_IF(vecC.empty());
  } else {
    // If the condition is a uniform scalar, we can just use it as is
    vecC.assign(packetWidth, cond);
  }

  auto name = Select->getName();
  for (unsigned i = 0; i < packetWidth; ++i) {
    results.push_back(B.CreateSelect(vecC[i], vecT[i], vecF[i], name));
  }
  return results;
}

Value *Packetizer::Impl::vectorizeReturn(ReturnInst *Return) {
  IRBuilder<> B(Return);
  Value *Op = packetize(Return->getOperand(0)).getAsValue();
  VECZ_FAIL_IF(!Op);
  IC.deleteInstructionLater(Return);
  return B.CreateRet(Op);
}

Value *Packetizer::Impl::vectorizeCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  VECZ_STAT_FAIL_IF(!Callee, VeczPacketizeFailCall);

  IRBuilder<> B(CI);
  // Handle LLVM intrinsics.
  if (Callee->isIntrinsic()) {
    Value *Result = nullptr;
    auto IntrID = Intrinsic::ID(Callee->getIntrinsicID());
    if (IntrID == Intrinsic::fmuladd || IntrID == Intrinsic::fma) {
      SmallVector<Value *, 3> Ops;
      SmallVector<Type *, 1> Tys;
      for (unsigned i = 0; i < 3; ++i) {
        Value *P = packetize(CI->getOperand(i)).getAsValue();
        VECZ_FAIL_IF(!P);
        Ops.push_back(P);
      }
      Tys.push_back(getWideType(CI->getType(), SimdWidth));
      Result = B.CreateIntrinsic(IntrID, Tys, Ops, CI, CI->getName());
    }

    if (Result) {
      IC.deleteInstructionLater(CI);
      return Result;
    }
  }

  // Handle internal builtins.
  if (Ctx.isInternalBuiltin(Callee)) {
    // These should have been handled by packetizeCall, if not, off to the
    // instantiator they go...
    if (auto MaskedOp = MemOp::get(CI, MemOpAccessKind::Masked)) {
      if (MaskedOp->isMaskedMemOp()) {
        return nullptr;
      }
    }
  }

  if (VectorizationContext::isVector(*CI)) {
    return nullptr;
  }

  // Handle external builtins.
  compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  auto const Builtin = BI.analyzeBuiltinCall(*CI, Dimension);

  if (Builtin.properties & compiler::utils::eBuiltinPropertyExecutionFlow) {
    return nullptr;
  }
  if (Builtin.properties & compiler::utils::eBuiltinPropertyWorkItem) {
    return vectorizeWorkGroupCall(CI, Builtin);
  }

  // Try to find a unit for this builtin.
  auto CalleeVec = Ctx.getVectorizedFunction(*Callee, SimdWidth);
  if (!CalleeVec) {
    // No vectorization strategy found. Fall back on Instantiation.
    return nullptr;
  }
  IC.deleteInstructionLater(CI);

  // Vectorize call operands.
  unsigned i = 0;
  AllocaInst *PointerRetAlloca = nullptr;
  Value *PointerRetAddr = nullptr;
  int PointerRetStride = 0;
  SmallVector<Value *, 4> Ops;
  for (const auto &TargetArg : CalleeVec.args) {
    // Handle scalar arguments.
    Value *ScalarOp = CI->getArgOperand(i);
    Type *ScalarTy = ScalarOp->getType();
    if (TargetArg.kind == VectorizationResult::Arg::POINTER_RETURN) {
      // 'Pointer return' arguments that are not sequential need to be handled
      // specially.
      auto *const PtrTy = dyn_cast<PointerType>(ScalarOp->getType());
      auto *const PtrEleTy = TargetArg.pointerRetPointeeTy;
      Value *Stride = SAR.buildMemoryStride(B, ScalarOp, PtrEleTy);
      VECZ_STAT_FAIL_IF(!Stride, VeczPacketizeFailStride);
      bool hasConstantStride = false;
      int64_t ConstantStride = 0;
      if (ConstantInt *CInt = dyn_cast<ConstantInt>(Stride)) {
        ConstantStride = CInt->getSExtValue();
        hasConstantStride = true;
      }
      VECZ_STAT_FAIL_IF(!hasConstantStride || ConstantStride < 1,
                        VeczPacketizeFailStride);
      if (ConstantStride == 1) {
        Ops.push_back(B.CreateBitCast(ScalarOp, TargetArg.type));
        i++;
        continue;
      }
      // Create an alloca in the function's entry block. The alloca will be
      // passed instead of the original pointer. After the function call,
      // the value from the alloca will be loaded sequentially and stored to the
      // original address using an interleaved store.
      VECZ_STAT_FAIL_IF(!PtrTy || PointerRetAddr, VeczPacketizeFailPtr);
      BasicBlock *BB = CI->getParent();
      VECZ_FAIL_IF(!BB);
      Function *F = BB->getParent();
      VECZ_FAIL_IF(!F);
      BasicBlock &EntryBB = F->getEntryBlock();
      B.SetInsertPoint(&*EntryBB.getFirstInsertionPt());
      Type *AllocaTy = getWideType(PtrEleTy, SimdWidth);
      PointerRetAlloca = B.CreateAlloca(AllocaTy, nullptr, "ptr_ret_temp");
      Value *NewOp = PointerRetAlloca;
      if (PtrTy->getAddressSpace() != 0) {
        Type *NewOpTy = PointerType::get(AllocaTy, PtrTy->getAddressSpace());
        NewOp = B.CreateAddrSpaceCast(NewOp, NewOpTy);
      }
      PointerRetAddr = ScalarOp;
      PointerRetStride = ConstantStride;
      Ops.push_back(NewOp);
      i++;
      continue;
    } else if (TargetArg.kind != VectorizationResult::Arg::VECTORIZED) {
      Ops.push_back(ScalarOp);
      i++;
      continue;
    }

    // Make sure the type is correct for vector arguments.
    auto VectorTy = dyn_cast<FixedVectorType>(TargetArg.type);
    VECZ_STAT_FAIL_IF(!VectorTy || VectorTy->getElementType() != ScalarTy,
                      VeczPacketizeFailType);

    // Vectorize scalar operands.
    Value *VecOp = packetize(ScalarOp).getAsValue();
    VECZ_FAIL_IF(!VecOp);
    Ops.push_back(VecOp);
    i++;
  }

  CallInst *NewCI = B.CreateCall(CalleeVec.get(), Ops, CI->getName());
  NewCI->setCallingConv(CI->getCallingConv());
  if (PointerRetAddr) {
    // Load the 'pointer return' value from the alloca and store it to the
    // original address using an interleaved store.
    LoadInst *PointerRetResult =
        B.CreateLoad(PointerRetAlloca->getAllocatedType(), PointerRetAlloca);
    Value *Stride = getSizeInt(B, PointerRetStride);
    auto *Store = createInterleavedStore(
        Ctx, PointerRetResult, PointerRetAddr, Stride,
        /*Mask*/ nullptr, /*EVL*/ nullptr, PointerRetAlloca->getAlign().value(),
        "", &*B.GetInsertPoint());
    if (!Store) {
      return nullptr;
    }
  }
  return NewCI;
}

Value *Packetizer::Impl::vectorizeWorkGroupCall(
    CallInst *CI, compiler::utils::BuiltinCall const &Builtin) {
  // Insert instructions after the call to the builtin, since they reference
  // the result of that call.
  IRBuilder<> B(buildAfter(CI, F));

  // Do not vectorize ranks equal to vectorization dimension. The value of
  // get_global_id with other ranks is uniform.

  Value *IDToSplat = CI;
  // Multiply the sub-group local ID by the vectorization factor, to vectorize
  // across the entire sub-group size.
  // For example, with a vector width of 4 and a mux sub-group size of 2, the
  // apparent sub-group size is 8 and the sub-group IDs are:
  // | mux sub group 0 | mux sub group 1 |
  // |-----------------|-----------------|
  // |  0   1   2   3  |  4   5   6   7  |
  if (Builtin.ID == compiler::utils::eMuxBuiltinGetSubGroupLocalId) {
    auto SimdWithAsVal = B.getInt32(SimdWidth.getKnownMinValue());
    IDToSplat = B.CreateMul(IDToSplat, !SimdWidth.isScalable()
                                           ? SimdWithAsVal
                                           : B.CreateVScale(SimdWithAsVal));
  }

  // Broadcast the builtin's return value.
  Value *Splat = B.CreateVectorSplat(SimdWidth, IDToSplat);

  // Add an index sequence [0, 1, 2, ...] to the value unless uniform.
  auto const Uniformity = Builtin.uniformity;
  if (Uniformity == compiler::utils::eBuiltinUniformityInstanceID ||
      Uniformity == compiler::utils::eBuiltinUniformityMaybeInstanceID) {
    Value *StepVector = multi_llvm::createIndexSequence(B, Splat->getType(),
                                                        SimdWidth, "index.vec");
    VECZ_FAIL_IF(!StepVector);

    Value *Result = B.CreateAdd(Splat, StepVector);

    if (Uniformity == compiler::utils::eBuiltinUniformityMaybeInstanceID) {
      Value *Rank = CI->getArgOperand(0);

      // if the Rank is varying, need to packetize that as well!
      if (UVR.isVarying(Rank)) {
        Rank = packetize(Rank).getAsValue();
        VECZ_FAIL_IF(!Rank);
      }
      Value *dim = ConstantInt::get(Rank->getType(), Dimension);
      Value *Test = B.CreateICmpEQ(Rank, dim);
      Result = B.CreateSelect(Test, Result, Splat, "maybe_rank");
    }
    return Result;
  } else if (Uniformity == compiler::utils::eBuiltinUniformityNever) {
    VECZ_FAIL();
  }

  return Splat;
}

Value *Packetizer::Impl::vectorizeAlloca(AllocaInst *alloca) {
  // We create an array allocation here, because the resulting value needs to
  // represent a vector of pointers, not a pointer to vector. As such, it's a
  // bit of a trick to handle scalable vectorization factors, since that would
  // require creating instrucions *before* the alloca, to get the array length,
  // which could be a surprise to some of our later passes that expect allocas
  // to be grouped at the top of the first Basic Block. This is not an LLVM
  // requirement, however, so it should be investigated.
  //
  // Note that normally, an alloca would not be packtized anyway, since access
  // is contiguous, Load and Store operations don't need to packetize their
  // pointer operand and the alloca would be widened after packetization, which
  // has no trouble with scalables. This function is required for the case that
  // some pointer-dependent instruction unexpectedly fails to packetize, and
  // falls back to instantiation, in which case we need a pointer per lane. In
  // actual fact, "normal" alloca vectorization is not very common, since such
  // allocas tend to be easy to remove by the Mem-to-Reg pass, so this "edge
  // case" is actually the most likely.
  //
  VECZ_FAIL_IF(SimdWidth.isScalable());
  unsigned fixedWidth = SimdWidth.getFixedValue();
  IRBuilder<> B(alloca);
  auto *const ty = alloca->getAllocatedType();
  AllocaInst *wideAlloca =
      B.CreateAlloca(ty, getSizeInt(B, fixedWidth), alloca->getName());
  wideAlloca->setAlignment(alloca->getAlign());

  // Put the GEP after all allocas.
  Instruction *insertPt = alloca;
  while (isa<AllocaInst>(*insertPt)) {
    insertPt = insertPt->getNextNonDebugInstruction();
  }
  B.SetInsertPoint(insertPt);
  deleteInstructionLater(alloca);

  auto *const idxTy = Ctx.dataLayout()->getIndexType(wideAlloca->getType());
  Value *const indices = multi_llvm::createIndexSequence(
      B, VectorType::get(idxTy, SimdWidth), SimdWidth);

  return B.CreateInBoundsGEP(ty, wideAlloca, ArrayRef<Value *>{indices},
                             Twine(alloca->getName(), ".lanes"));
}

Value *Packetizer::Impl::vectorizeExtractValue(ExtractValueInst *ExtractValue) {
  IRBuilder<> B(buildAfter(ExtractValue, F));

  Value *Aggregate =
      packetize(ExtractValue->getAggregateOperand()).getAsValue();
  SmallVector<unsigned, 4> Indices;
  Indices.push_back(0);
  for (auto Index : ExtractValue->indices()) {
    Indices.push_back(Index);
  }

  SmallVector<Value *, 16> Extracts;

  VECZ_FAIL_IF(SimdWidth.isScalable());
  auto Width = SimdWidth.getFixedValue();

  // Check that the width is non-zero so the zeroth element is initialized.
  VECZ_FAIL_IF(Width < 1);

  for (decltype(Width) i = 0; i < Width; i++) {
    Indices[0] = i;
    Extracts.push_back(B.CreateExtractValue(Aggregate, Indices));
  }

  Type *CompositeTy = getWideType(Extracts[0]->getType(), SimdWidth);
  Value *Result = UndefValue::get(CompositeTy);
  for (decltype(Width) i = 0; i < Width; i++) {
    Result = B.CreateInsertElement(Result, Extracts[i], B.getInt32(i));
  }

  return Result;
}

ValuePacket Packetizer::Impl::packetizeInsertElement(
    InsertElementInst *InsertElement) {
  ValuePacket results;
  Value *Result = nullptr;

  Value *Into = InsertElement->getOperand(0);
  assert(Into && "Could not get operand 0 of InsertElement");
  const auto ScalarWidth = multi_llvm::getVectorNumElements(Into->getType());

  Value *Elt = InsertElement->getOperand(1);
  Value *Index = InsertElement->getOperand(2);
  assert(Elt && "Could not get operand 1 of InsertElement");
  assert(Index && "Could not get operand 2 of InsertElement");

  if (SimdWidth.isScalable()) {
    auto packetWidth = getPacketWidthForType(Into->getType());
    auto intoVals = packetizeAndGet(Into, packetWidth);
    // Scalable vectorization (currently) only ever generates 1 packet
    PACK_FAIL_IF(intoVals.size() != 1);
    Value *packetizedInto = intoVals.front();

    auto eltPacketWidth = getPacketWidthForType(Elt->getType());
    auto eltVals = packetizeAndGet(Elt, eltPacketWidth);
    // Scalable vectorization (currently) only ever generates 1 packet
    PACK_FAIL_IF(eltVals.size() != 1);
    Value *packetizedElt = eltVals.front();

    Value *packetizedIndices = packetizeIfVarying(Index);

    auto *packetizedEltTy = packetizedElt->getType();
    auto *packetizedIntoTy = packetizedInto->getType();
    auto *scalarTy = packetizedEltTy->getScalarType();

    // Compiler support for masked.gather/riscv.vrgather* on i1 vectors is
    // lacking, so emit this operation as the equivalent i8 vector instead.
    auto *const origPacketizedIntoTy = packetizedIntoTy;
    const bool upcastI1AsI8 = scalarTy->isIntegerTy(1);
    IRBuilder<> B(buildAfter(InsertElement, F));
    if (upcastI1AsI8) {
      auto *const int8Ty = Type::getInt8Ty(F.getContext());
      packetizedIntoTy = llvm::VectorType::get(
          int8Ty, multi_llvm::getVectorElementCount(packetizedIntoTy));
      packetizedEltTy = llvm::VectorType::get(
          int8Ty, multi_llvm::getVectorElementCount(packetizedEltTy));
      packetizedElt = B.CreateSExt(packetizedElt, packetizedEltTy);
      packetizedInto = B.CreateSExt(packetizedInto, packetizedIntoTy);
    }

    // If we're vector predicating, scale the vector length up by the original
    // number of vector elements.
    auto *const EVL = VL ? B.CreateMul(VL, B.getInt32(ScalarWidth)) : nullptr;

    auto *packetizedInsert = Ctx.targetInfo().createScalableInsertElement(
        B, Ctx, InsertElement, packetizedElt, packetizedInto, packetizedIndices,
        EVL);

    // If we've been performing this broadcast as i8, now's the time to
    // truncate back down to i1
    if (upcastI1AsI8) {
      packetizedInsert = B.CreateTrunc(packetizedInsert, origPacketizedIntoTy);
    }

    IC.deleteInstructionLater(InsertElement);
    results.push_back(packetizedInsert);
    return results;
  }

  auto Width = SimdWidth.getFixedValue();

  IRBuilder<> B(buildAfter(InsertElement, F));

  const auto Name = InsertElement->getName();
  if (auto *CIndex = dyn_cast<ConstantInt>(Index)) {
    auto IdxVal = CIndex->getZExtValue();

    auto packetWidth = getPacketWidthForType(Into->getType());
    PACK_FAIL_IF(packetWidth == Width);

    auto Intos = packetizeAndGet(Into, packetWidth);
    PACK_FAIL_IF(Intos.empty());

    auto res = packetize(Elt);
    PACK_FAIL_IF(!res);

    if (res.info->numInstances == 0) {
      // If the element was broadcast, it's better just to create more insert
      // element instructions..
      const auto instanceWidth =
          multi_llvm::getVectorNumElements(Intos.front()->getType());
      for (unsigned i = 0; i < packetWidth; ++i) {
        results.push_back(Intos[i]);
        for (unsigned j = IdxVal; j < instanceWidth; j += ScalarWidth) {
          results.back() =
              B.CreateInsertElement(results.back(), Elt, B.getInt32(j), Name);
        }
      }
      return results;
    }

    SmallVector<Value *, 16> Elts;
    res.getPacketValues(packetWidth, Elts);
    PACK_FAIL_IF(Elts.empty());

    const auto *VecTy = cast<FixedVectorType>(Intos.front()->getType());
    const unsigned VecWidth = VecTy->getNumElements();
    PACK_FAIL_IF(VecWidth == ScalarWidth);
    {
      // Can only shuffle two vectors of the same size, so redistribute
      // the packetized elements vector
      SmallVector<int, 16> Mask;
      for (size_t i = 0; i < VecWidth; ++i) {
        Mask.push_back(i / ScalarWidth);
      }

      auto *Undef = UndefValue::get(Elts.front()->getType());
      for (unsigned i = 0; i < packetWidth; ++i) {
        results.push_back(createOptimalShuffle(B, Elts[i], Undef, Mask, Name));
      }
    }
    if (isa<UndefValue>(Into)) {
      // Inserting into nothing so we can just use it as is..
      return results;
    } else {
      SmallVector<int, 16> Mask;
      for (size_t i = 0; i < VecWidth; ++i) {
        int j = VecWidth + i;
        if (i == IdxVal) {
          j = i;
          IdxVal += ScalarWidth;
        }
        Mask.push_back(j);
      }

      for (unsigned i = 0; i < packetWidth; ++i) {
        results[i] = createOptimalShuffle(B, results[i], Intos[i], Mask, Name);
      }
      return results;
    }
  } else {
    Into = packetize(Into).getAsValue();
    PACK_FAIL_IF(!Into);
    Value *Elts = packetizeIfVarying(Elt);
    PACK_FAIL_IF(!Elts);
    Value *Indices = packetizeIfVarying(Index);
    PACK_FAIL_IF(!Indices);

    Result = Into;
    if (Indices != Index) {
      Type *IdxTy = Index->getType();
      SmallVector<Constant *, 16> Offsets;
      for (unsigned i = 0; i < Width; ++i) {
        Offsets.push_back(ConstantInt::get(IdxTy, i * ScalarWidth));
      }
      Value *Add = B.CreateAdd(Indices, ConstantVector::get(Offsets));

      for (unsigned i = 0; i < Width; ++i) {
        Value *ExtractElt =
            (Elts != Elt) ? B.CreateExtractElement(Elts, B.getInt32(i)) : Elt;
        Value *ExtractIdx = B.CreateExtractElement(Add, B.getInt32(i));
        Result = B.CreateInsertElement(Result, ExtractElt, ExtractIdx, Name);
      }
    } else {
      for (unsigned i = 0; i < Width; ++i) {
        Value *ExtractElt =
            (Elts != Elt) ? B.CreateExtractElement(Elts, B.getInt32(i)) : Elt;
        Value *InsertIdx = B.CreateAdd(Index, B.getInt32(i * ScalarWidth));
        Result = B.CreateInsertElement(Result, ExtractElt, InsertIdx, Name);
      }
    }
  }
  IC.deleteInstructionLater(InsertElement);
  results.push_back(Result);
  return results;
}

ValuePacket Packetizer::Impl::packetizeExtractElement(
    ExtractElementInst *ExtractElement) {
  ValuePacket results;
  Value *Result = nullptr;

  Value *Src = ExtractElement->getOperand(0);
  Value *Index = ExtractElement->getOperand(1);
  assert(Src && "Could not get operand 0 of ExtractElement");
  assert(Index && "Could not get operand 1 of ExtractElement");

  if (SimdWidth.isScalable()) {
    auto packetWidth = getPacketWidthForType(Src->getType());
    auto srcVals = packetizeAndGet(Src, packetWidth);
    // Scalable vectorization (currently) only ever generates 1 packet
    PACK_FAIL_IF(srcVals.size() != 1);
    Value *packetizedSrc = srcVals.front();

    Value *packetizedIndices = packetizeIfVarying(Index);

    Value *packetizedExtract = [&]() {
      IRBuilder<> B(buildAfter(ExtractElement, F));

      auto *narrowTy = getWideType(ExtractElement->getType(), SimdWidth);
      auto *const origNarrowTy = narrowTy;
      auto *origSrc = ExtractElement->getOperand(0);
      auto *origTy = origSrc->getType();
      auto *eltTy = origTy->getScalarType()->getScalarType();

      // Compiler support for masked.gather/riscv.vrgather* on i1
      // vectors is lacking, so emit this operation as the equivalent
      // i8 vector instead.
      const bool upcastI1AsI8 = eltTy->isIntegerTy(/*BitWidth*/ 1);
      if (upcastI1AsI8) {
        auto *const int8Ty = B.getInt8Ty();
        auto *wideTy = llvm::VectorType::get(
            int8Ty,
            multi_llvm::getVectorElementCount(packetizedSrc->getType()));
        narrowTy = llvm::VectorType::get(
            int8Ty, multi_llvm::getVectorElementCount(narrowTy));
        packetizedSrc = B.CreateSExt(packetizedSrc, wideTy);
      }

      Value *extract = Ctx.targetInfo().createScalableExtractElement(
          B, Ctx, ExtractElement, narrowTy, packetizedSrc, packetizedIndices,
          VL);

      // If we've been performing this broadcast as i8, now's the time to
      // truncate back down to i1
      if (extract && upcastI1AsI8) {
        extract = B.CreateTrunc(extract, origNarrowTy);
      }

      return extract;
    }();
    PACK_FAIL_IF(!packetizedExtract);

    IC.deleteInstructionLater(ExtractElement);
    results.push_back(packetizedExtract);
    return results;
  }

  auto Width = SimdWidth.getFixedValue();

  const auto ScalarWidth = multi_llvm::getVectorNumElements(Src->getType());

  IRBuilder<> B(buildAfter(ExtractElement, F));
  const auto Name = ExtractElement->getName();
  if (auto *CIndex = dyn_cast<ConstantInt>(Index)) {
    auto IdxVal = CIndex->getZExtValue();

    auto packetWidth = getPacketWidthForType(ExtractElement->getType());
    auto srcVals = packetizeAndGet(Src, packetWidth);
    PACK_FAIL_IF(srcVals.empty());

    auto resultWidth = Width / packetWidth;
    if (packetWidth == 1) {
      srcVals.push_back(UndefValue::get(srcVals.front()->getType()));
    } else {
      resultWidth *= 2;
    }

    SmallVector<int, 16> Mask;
    for (size_t i = 0, j = IdxVal; i < resultWidth; ++i, j += ScalarWidth) {
      Mask.push_back(j);
    }

    for (unsigned i = 0; i < packetWidth; i += 2) {
      results.push_back(
          createOptimalShuffle(B, srcVals[i], srcVals[i + 1], Mask, Name));
    }
    return results;
  } else {
    Value *Sources = packetizeIfVarying(Src);
    PACK_FAIL_IF(!Sources);
    Value *Indices = packetizeIfVarying(Index);
    PACK_FAIL_IF(!Indices);

    Result = UndefValue::get(getWideType(ExtractElement->getType(), SimdWidth));
    if (Indices != Index) {
      Type *IdxTy = Index->getType();
      SmallVector<Constant *, 16> Offsets;
      for (unsigned i = 0; i < Width; ++i) {
        Offsets.push_back(ConstantInt::get(IdxTy, i * ScalarWidth));
      }

      if (Sources != Src) {
        Indices = B.CreateAdd(Indices, ConstantVector::get(Offsets));
      }

      for (unsigned i = 0; i < Width; ++i) {
        Value *ExtractIdx = B.CreateExtractElement(Indices, B.getInt32(i));
        Value *ExtractElt = B.CreateExtractElement(Sources, ExtractIdx);
        Result = B.CreateInsertElement(Result, ExtractElt, B.getInt32(i), Name);
      }
    } else {
      for (unsigned i = 0, j = 0; i < Width; ++i, j += ScalarWidth) {
        Value *ExtractIdx = (Sources != Src && i != 0)
                                ? B.CreateAdd(Index, B.getInt32(j))
                                : Index;
        Value *ExtractElt = B.CreateExtractElement(Sources, ExtractIdx);
        Result = B.CreateInsertElement(Result, ExtractElt, B.getInt32(i), Name);
      }
    }
  }
  IC.deleteInstructionLater(ExtractElement);
  results.push_back(Result);
  return results;
}

ValuePacket Packetizer::Impl::packetizeShuffleVector(
    ShuffleVectorInst *Shuffle) {
  Value *const srcA = Shuffle->getOperand(0);
  Value *const srcB = Shuffle->getOperand(1);
  assert(srcA && "Could not get operand 0 from Shuffle");
  assert(srcB && "Could not get operand 1 from Shuffle");
  auto *const ty = Shuffle->getType();
  auto *const tyA = srcA->getType();
  auto packetWidth =
      std::max(getPacketWidthForType(ty), getPacketWidthForType(tyA));

  ValuePacket results;
  IRBuilder<> B(buildAfter(Shuffle, F));
  auto const scalarWidth = multi_llvm::getVectorNumElements(tyA);

  if (SimdWidth.isScalable()) {
    PACK_FAIL_IF(packetWidth != 1);
    if (auto *const SplatVal = getSplatValue(Shuffle)) {
      // Handle splats as a special case.
      auto Splats = packetizeAndGet(SplatVal);
      PACK_FAIL_IF(!createSubSplats(Ctx.targetInfo(), B, Splats, scalarWidth));
      return Splats;
    } else {
      // It isn't safe to do it if it's not a power of 2.
      PACK_FAIL_IF(!isPowerOf2_32(scalarWidth));
      TargetInfo &VTI = Ctx.targetInfo();

      auto const dstScalarWidth = multi_llvm::getVectorNumElements(ty);
      auto const fullWidth = SimdWidth * dstScalarWidth;

      // If we're vector-predicating a vector access, scale the vector length
      // up by the original number of vector elements.
      auto *const EVL =
          VL ? B.CreateMul(VL, B.getInt32(dstScalarWidth)) : nullptr;

      auto *const mask = Shuffle->getShuffleMaskForBitcode();
      auto *const vecMask =
          VTI.createOuterScalableBroadcast(B, mask, EVL, SimdWidth);

      auto *const idxVector = multi_llvm::createIndexSequence(
          B, VectorType::get(B.getInt32Ty(), fullWidth), fullWidth);

      // We need to create offsets into the source operand subvectors, to add
      // onto the broadcast shuffle mask, so that each subvector of the
      // destination indices into the corresponding subvector of the source.
      // That is, for a source vector width of `n` we need the indices
      // `[0, n, 2*n, 3*n ...]`, which correspond to the indices of the first
      // element of each subvector of the packetized source. For a destination
      // vector of width `m` we need `m` instances of each index.
      //
      // We can compute the offset vector as `offset[i] = floor(i / m) * n`.
      Value *offset = nullptr;
      if (dstScalarWidth == scalarWidth) {
        // If the source and destination are the same size, we have a special
        // case and can mask off the LSBs of the index vector instead. i.e.
        //     `offset[i] = i & -n`
        // For instance, for `n == 4` we have offset indices:
        // [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, ... ].
        offset = B.CreateAnd(
            idxVector,
            ConstantVector::getSplat(fullWidth, B.getInt32(-scalarWidth)));
      } else {
        auto *const subVecID = B.CreateUDiv(
            idxVector,
            ConstantVector::getSplat(fullWidth, B.getInt32(dstScalarWidth)));
        offset = B.CreateMul(subVecID, ConstantVector::getSplat(
                                           fullWidth, B.getInt32(scalarWidth)));
      }

      auto *const vecA = packetizeAndGet(srcA, 1).front();
      if (isa<UndefValue>(srcB)) {
        auto *const adjust = B.CreateAdd(vecMask, offset, "shuffleMask");
        auto *const shuffleA = VTI.createVectorShuffle(B, vecA, adjust, EVL);
        results.push_back(shuffleA);
      } else {
        // For a two-source shuffle, we shuffle each source separately and then
        // select between the results. It might sound tempting to concatenate
        // the sources first and use a single shuffle, but since the results
        // need to be interleaved, it makes the mask computation somewhat more
        // complicated, with indices dependent on the vector scale factor.
        auto *const vecB = packetizeAndGet(srcB, 1).front();

        auto *const whichCmp = B.CreateICmpUGE(
            vecMask,
            ConstantVector::getSplat(fullWidth, B.getInt32(scalarWidth)));
        auto *const safeMask = B.CreateAnd(
            vecMask,
            ConstantVector::getSplat(fullWidth, B.getInt32(scalarWidth - 1)));

        auto *const adjust = B.CreateAdd(safeMask, offset, "shuffleMask");
        auto *const shuffleA = VTI.createVectorShuffle(B, vecA, adjust, EVL);
        auto *const shuffleB = VTI.createVectorShuffle(B, vecB, adjust, EVL);
        results.push_back(B.CreateSelect(whichCmp, shuffleB, shuffleA));
      }

      return results;
    }
  }

  auto srcsA = packetizeAndGet(srcA, packetWidth);
  auto srcsB = packetizeAndGet(srcB, packetWidth);
  PACK_FAIL_IF(srcsA.empty() || srcsB.empty());

  auto width = SimdWidth.getFixedValue() / packetWidth;

  // Because up to and including LLVM 10, the IR Builder accepts a mask as a
  // vector of uint32_t, but getShuffleMask returns an array of ints. So
  // we do it this way.
  const auto &origMask = Shuffle->getShuffleMask();
  SmallVector<int, 16> mask(origMask.begin(), origMask.end());

  // Adjust any indices that select from the second source vector
  const auto adjust =
      isa<UndefValue>(srcB) ? -scalarWidth : (width - 1) * scalarWidth;
  for (auto &idx : mask) {
    if (idx != int(-1) && idx >= int(scalarWidth)) {
      idx += adjust;
    }
  }

  // Duplicate the mask over the vectorized width
  const auto size = mask.size();
  mask.reserve(size * width);
  for (unsigned i = 1, k = 0; i < width; ++i, k += size) {
    for (unsigned j = 0; j < size; ++j) {
      auto maskElem = mask[k + j];
      if (maskElem != int(-1)) {
        maskElem += scalarWidth;
      }
      mask.push_back(maskElem);
    }
  }

  const auto name = Shuffle->getName();
  for (unsigned i = 0; i < packetWidth; ++i) {
    results.push_back(createOptimalShuffle(B, srcsA[i], srcsB[i], mask, name));
  }
  return results;
}
