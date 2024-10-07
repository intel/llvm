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

#include "transform/control_flow_conversion_pass.h"

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/mangling.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/basicblock_helper.h>

#include <queue>
#include <utility>

#include "analysis/control_flow_analysis.h"
#include "analysis/divergence_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "control_flow_boscc.h"
#include "control_flow_roscc.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "reachability.h"
#include "transform/passes.h"
#include "vecz/vecz_choices.h"

#define DEBUG_TYPE "vecz-cf"

using namespace llvm;
using namespace vecz;

class ControlFlowConversionState::Impl : public ControlFlowConversionState {
 public:
  Impl(Function &F, FunctionAnalysisManager &AM)
      : ControlFlowConversionState(F, AM) {}

  PreservedAnalyses run(Function &, FunctionAnalysisManager &);

 private:
  /// @brief utility struct used by LinearizeCFG to allow block retargeting
  /// info to be stored in a single contiguous vector of variable-length
  /// subvectors. This avoids having to use a vector of vectors, and all
  /// the individual heap allocations that would involve. Empirically (based on
  /// UnitCL) we have approximately one new target per Basic Block overall,
  /// and never more than 2 (which is not to say more than 2 is impossible).
  /// Since we iterate over all NewTargetInfos linearly, we only need to record
  /// the number of targets for each block, and not their starting indices.
  struct Linearization {
    struct NewTargetInfo {
      BasicBlock *BB;
      size_t numTargets = 0;

      NewTargetInfo(BasicBlock *bb) : BB(bb) {}
    };

    std::vector<NewTargetInfo> infos;
    std::vector<BasicBlock *> data;

    void beginBlock(BasicBlock *BB) { infos.emplace_back(BB); }
    size_t currentSize() const { return infos.back().numTargets; }
    void push(BasicBlock *BB) {
      data.push_back(BB);
      ++infos.back().numTargets;
    }
  };

  /// @brief Type that maps exit blocks to exit mask information.
  using DenseExitPHIMap = SmallDenseMap<const BasicBlock *, PHINode *, 2>;
  /// @brief Type that maps exiting blocks to update mask information.
  using DenseExitUpdateMap =
      SmallDenseMap<const BasicBlock *, BinaryOperator *, 2>;

  struct LoopMasksInfo {
    /// @brief Keep track of which instances left the loop through which exit
    ///        (persisted throughout the whole loop).
    DenseExitPHIMap persistedDivergentExitMasks;
    /// @brief Divergent loop exit masks updated for the current iteration.
    DenseExitUpdateMap updatedPersistedDivergentExitMasks;
    /// @brief Combined divergent loop exit masks of the current iteration.
    Instruction *combinedDivergentExitMask = nullptr;
    /// @brief Combined divergent loop exit masks of the whole loop.
    Instruction *persistedCombinedDivergentExitMask = nullptr;
  };

  /// @brief Convert the function's CFG to data-flow.
  /// @return true if the function's CFG was converted, false otherwise.
  bool convertToDataFlow();

  /// @brief Generate masks needed to do control-flow to data-flow conversion.
  /// @return true if masks were generated successfully, false otherwise.
  bool generateMasks();

  /// @brief Generate masks for the given block.
  /// @param[in] BB Block whose masks we are generating.
  /// @return true if no problem occurred, false otherwise.
  bool createMasks(BasicBlock &BB);

  /// @brief Create entry mask for the given block.
  /// @param[in] BB Block whose masks we are generating.
  /// @return true if no problem occurred, false otherwise.
  bool createEntryMasks(BasicBlock &BB);

  /// @brief Create exit mask for the given block.
  /// @param[in] BB Block whose masks we are generating.
  /// @param[in] isBOSCCEntry Whether BB creates a uniform region.
  /// @return true if no problem occurred, false otherwise.
  bool createExitMasks(BasicBlock &BB, bool isBOSCCEntry = false);

  /// @brief Create loop exit masks for the given loop.
  /// @param[in,out] LTag Information on the loop we are evaluating.
  /// @return true if no problem occurred, false otherwise.
  bool createLoopExitMasks(LoopTag &LTag);

  /// @brief Combine all information about instances that left the loop in the
  ///        current iteration.
  /// @param[in,out] LTag Information on the loop we are evaluating.
  /// @return true if no problem occurred, false otherwise.
  bool createCombinedLoopExitMask(LoopTag &LTag);

  /// @brief Apply masks to basic blocks in the function, to prevent
  /// side-effects for inactive instances.
  ///
  /// @return llvm::Error::success if masks were applied successfully, an error
  /// message explaining the failure otherwise.
  Error applyMasks();

  /// @brief Apply a mask to the given basic block, to prevent side-effects for
  /// inactive instances.
  ///
  /// @param[in] BB Basic block to apply masks to.
  /// @param[in] mask Mask to apply.
  ///
  /// @return llvm::Error::success if masks were applied successfully, an error
  /// message explaining the failure otherwise.
  Error applyMask(BasicBlock &BB, Value *mask);

  /// @brief Emit a call instructions to the masked version of the called
  /// function.
  ///
  /// @param[in] CI The call instructions to create a masked version of
  /// @param[in] entryBit The Value that determines if the lane is active or
  /// not.
  /// @return The call instruction to the masked version.
  CallInst *emitMaskedVersion(CallInst *CI, Value *entryBit);

  /// @brief Create a masked version of the given function
  ///
  /// The Function (F) to be masked will be extracted from the CallInst and a
  /// new Function (NewFunction) will be generated. NewFunction takes the same
  /// arguments as F, plus an additional boolean argument that determines if the
  /// lane is active or not. If the boolean argument is true, then NewFunction
  /// will execute F and (if it's not void) return its return value. Vararg
  /// functions are supported by expanding their arguments.
  ///
  /// @param[in] CI The call instructions to create a masked version of
  /// @return The masked function
  Function *getOrCreateMaskedVersion(CallInst *CI);

  /// @brief a type that maps unmasked instructions onto masked replacements.
  using DeletionMap = SmallVector<std::pair<Instruction *, Value *>, 4>;

  /// @brief Attempt to apply a mask to an Instruction as a Memory Operation
  ///
  /// @param[in] I The Binary Operation to apply the mask to
  /// @param[in] mask The mask to apply to the MemOp
  /// @param[out] toDelete mapping of deleted unmasked operations
  /// @param[out] safeDivisors a cache of re-usable known non-zero divisors
  /// @return true if it was a BinOp, false otherwise
  bool tryApplyMaskToBinOp(Instruction &I, Value *mask, DeletionMap &toDelete,
                           DenseMap<Value *, Value *> &safeDivisors);

  /// @brief Attempt to apply a mask to a Memory Operation
  ///
  /// @param[in] op The MemOp to apply the mask to
  /// @param[in] mask The mask to apply to the MemOp
  /// @param[out] toDelete mapping of deleted unmasked operations
  /// @return true of the MemOp got masked, false otherwise
  bool tryApplyMaskToMemOp(MemOp &op, Value *mask, DeletionMap &toDelete);

  /// @brief Attempt to apply a mask to an Instruction as a Memory Operation
  ///
  /// @param[in] CI The call instruction to apply the mask to
  /// @param[in] mask The mask to apply to the MemOp
  /// @param[out] toDelete mapping of deleted unmasked operations
  /// @return true if it is valid to mask this call, false otherwise
  bool applyMaskToCall(CallInst *CI, Value *mask, DeletionMap &toDelete);

  /// @brief Attempt to apply a mask to an atomic instruction via a builtin
  /// call.
  ///
  /// @param[in] I The (atomic) instruction to apply the mask to
  /// @param[in] mask The mask to apply to the masked atomic
  /// @param[out] toDelete mapping of deleted unmasked operations
  /// @return true if it is valid to mask this atomic, false otherwise
  bool applyMaskToAtomic(Instruction &I, Value *mask, DeletionMap &toDelete);

  /// @brief Linearize a CFG.
  /// @return true if no problem occurred, false otherwise.
  bool partiallyLinearizeCFG();

  /// @brief Create the reduction functions needed to vectorize the branch
  /// @return true on success, false otherwise
  bool createBranchReductions();

  /// @brief Uniformize every divergent loop.
  ///
  /// @return true if no problem occurred, false otherwise.
  bool uniformizeDivergentLoops();

  /// @brief Assign a divergent loop a single loop exit from which all other
  ///        exits will be rewired.
  /// @param[in] LTag Tag of the processed loop
  /// @return true if no problem occurred, false otherwise.
  bool computeDivergentLoopPureExit(LoopTag &LTag);

  /// @brief Rewire every loop exit block such that the loop can be considered
  ///        uniform.
  ///
  /// @param[in] LTag Tag of the processed loop
  /// @param[in] exitBlocks List of exit blocks before any transformation
  /// @return true if no problem occurred, false otherwise.
  bool rewireDivergentLoopExitBlocks(
      LoopTag &LTag, const SmallVectorImpl<BasicBlock *> &exitBlocks);

  /// @brief Generate blend operations to discard execution of inactive
  /// instances.
  /// @param[in] LTag The loop whose live value is being handled.
  /// @return true if no problem occurred, false otherwise.
  bool generateDivergentLoopResults(LoopTag &LTag);

  /// @brief Generate loop live value update instructions.
  /// @param[in] LLV   The loop live value we want to generate instructions for.
  /// @param[in] LTag The loop whose live value is being handled.
  /// @return true if no problem occurred, false otherwise.
  bool generateDivergentLoopResultUpdates(Value *LLV, LoopTag &LTag);

  /// @brief Generate blend instruction for loop live values at the latch.
  /// @param[in] LTag The loop whose live values are being handled.
  /// @param[in] exitBlocks List of exit blocks before any transformation
  /// @return true if no problem occurred, false otherwise.
  bool blendDivergentLoopLiveValues(
      LoopTag &LTag, const SmallVectorImpl<BasicBlock *> &exitBlocks);

  /// @brief Generate blend instruction for loop exit masks at the latch.
  ///
  /// @param[in] LTag Tag of the processed loop
  /// @param[in] exitEdges List of exit edges before any transformation
  /// @param[in] exitBlocks List of exit blocks before any transformation
  /// @return true if no problem occurred, false otherwise.
  bool blendDivergentLoopExitMasks(
      LoopTag &LTag, const SmallVectorImpl<Loop::Edge> &exitEdges,
      const SmallVectorImpl<BasicBlock *> &exitBlocks);

  /// @brief Replace uses of loop values outside of a divergent loop.
  ///
  /// @param[in] LTag Tag of the processed loop
  /// @param[in] from Instruction to be replaced.
  /// @param[in] to Instruction to replace `from` with.
  /// @param[in] exitBlocks Exit blocks of the loop.
  /// @return true if no problem occurred, false otherwise.
  bool replaceUsesOutsideDivergentLoop(
      LoopTag &LTag, Value *from, Value *to,
      const SmallVectorImpl<BasicBlock *> &exitBlocks);

  /// @brief Assign new targets to edges based on the dominance-compact
  ///        ordering.
  /// @param[out] lin New target information for each BasicBlock
  /// @return true if no problem occurred, false otherwise.
  bool computeNewTargets(Linearization &lin);

  /// @brief Linearize the CFG with the new calculated edges.
  /// @return true if no problem occurred, false otherwise.
  bool linearizeCFG();

  /// @brief Generate blend operations to discard execution of inactive
  /// instances.
  /// @return true if no problem occurred, false otherwise.
  bool generateSelects();

  /// @brief Split a phi instruction into several select instructions.
  /// @param[in,out] PHI The PHI node we want to split.
  /// @param[in]     B  The block PHI belongs to.
  /// @return true if no problem occurred, false otherwise.
  bool generateSelectFromPHI(PHINode *PHI, BasicBlock *B);

  /// @brief Repair the SSA form. First blend and create new masks from the
  ///        new wires, then blend all the instructions that need blending.
  /// @return true if no errors occurred.
  bool repairSSA();

  /// @brief Update the incoming blocks of phi nodes whose predecessors have
  ///        changed whilst rewiring.
  /// @return true if no errors occurred.
  bool updatePHIsIncomings();

  /// @brief Blend instructions before their uses if divergence happened
  ///        inbetween.
  /// @return true if no errors occurred.
  bool blendInstructions();

  /// @brief Simplify the mask instructions.
  /// @return true if no errors occurred.
  bool simplifyMasks();

  /// @brief Check all blocks have a unique index order.
  /// @return true if no errors occurred.
  bool checkBlocksOrder() const;

  /// @brief Upon modifying a mask, we need to update the in-memory masks as
  ///        well.
  /// @param[in] src The block whose mask changed
  /// @param[in] from The old mask
  /// @param[in] to The new mask
  void replaceMasks(BasicBlock *src, Value *from, Value *to);

  /// @brief Upon removing an instruction, we need to also update our internal
  ///        containers.
  /// @param[in] from The old value
  /// @param[in] to The new value
  void updateMaps(Value *from, Value *to);

  BasicBlock *functionExitBlock = nullptr;
  DenseSet<const Instruction *> blends;
  DenseMap<Loop *, LoopMasksInfo> LoopMasks;
};

STATISTIC(VeczCFGFail,
          "Number of kernels that failed control flow conversion [ID#L80]");

// Set this to enable all-of masks in the latch of divergent loops. This can
// be interesting if there exists an intrinsic that, when comparing vector
// instructions, can immediately stop comparing if one of the operands if false.
// In counterpart, this makes us update two more values per divergent loops
// (said values allowing to keep track of which instances left the loop).
//
// Because no such intrinsic exists to my knowledge, we don't set this by
// default.
#undef ALL_OF_DIVERGENT_LOOP_LATCH

namespace {

Instruction *getInsertionPt(BasicBlock &BB) {
  // We have to insert instructions after any Allocas
  auto it = BB.getFirstInsertionPt();
  while (isa<AllocaInst>(*it)) {
    ++it;
  }
  return &*it;
}

Instruction *copyMask(Value *mask, Twine name, Instruction *insertBefore) {
  VECZ_ERROR_IF(!mask || !insertBefore,
                "Trying to copy mask with invalid arguments");
  return BinaryOperator::CreateAnd(mask, getDefaultValue(mask->getType(), 1),
                                   name, insertBefore);
}

Instruction *copyEntryMask(Value *mask, BasicBlock &BB) {
  VECZ_ERROR_IF(!mask, "Trying to copy entry mask with invalid arguments");
  return copyMask(mask, BB.getName() + ".entry_mask", getInsertionPt(BB));
}

Instruction *copyExitMask(Value *mask, StringRef base, BasicBlock &BB) {
  VECZ_ERROR_IF(!mask, "Trying to copy exit mask with invalid arguments");
  return copyMask(mask, base + ".exit_mask", BB.getTerminator());
}

/// Wrap a string into an llvm::StringError, pointing to an instruction.
static inline Error makeStringError(const Twine &message, Instruction &I) {
  std::string helper_str = message.str();
  raw_string_ostream helper_stream(helper_str);
  helper_stream << " " << I;
  return make_error<StringError>(helper_stream.str(), inconvertibleErrorCode());
}

// A helper method to determine whether a branch condition
// (expected to be an i1 result of a comparison instruction) is truly uniform.
static bool isBranchCondTrulyUniform(Value *cond, UniformValueResult &UVR) {
  const auto *cmp = dyn_cast_if_present<CmpInst>(cond);
  if (!cmp || cmp->getType()->isVectorTy()) {
    return false;
  }

  return UVR.isTrueUniform(cmp);
}
}  // namespace

////////////////////////////////////////////////////////////////////////////////

char ControlFlowConversionPass::PassID = 0;

PreservedAnalyses ControlFlowConversionPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  ControlFlowConversionState::Impl state(F, AM);
  return state.run(F, AM);
}

ControlFlowConversionState::ControlFlowConversionState(
    Function &F, FunctionAnalysisManager &AM)
    : F(F),
      AM(AM),
      VU(AM.getResult<VectorizationUnitAnalysis>(F).getVU()),
      Ctx(AM.getResult<VectorizationContextAnalysis>(F).getContext()) {}

PreservedAnalyses ControlFlowConversionState::Impl::run(
    Function &F, FunctionAnalysisManager &AM) {
  const auto &CFGR = AM.getResult<CFGAnalysis>(F);
  if (CFGR.getFailed()) {
    ++VeczCFGFail;
    return VU.setFailed("Cannot vectorize the CFG for", &F, &F);
  } else if (!CFGR.isConversionNeeded()) {
    return PreservedAnalyses::all();
  }
  functionExitBlock = CFGR.getExitBlock();

  if (!convertToDataFlow()) {
    // This pass may leave the function in an invalid state. Instead of doing
    // so, and hoping that later passes don't throw verification failures back
    // at us, replace the function body with an unreachable statement. Marking
    // vectorization has having failed will mean the function will later be
    // deleted.
    // Note that this is quite coarse-grained; we could be cleverer, e.g., by
    // returning whether convertToDataFlow has (potentially) left behind an
    // invalid function.
    ++VeczCFGFail;
    VU.setFailed("Control flow conversion failed for", &F, VU.scalarFunction());
    F.deleteBody();
    BasicBlock *BB = BasicBlock::Create(F.getContext(), "entry", &F);
    IRBuilder<> IRB(BB);
    IRB.CreateUnreachable();
    return PreservedAnalyses::none();
  }

  PreservedAnalyses Preserved;
  Preserved.preserve<DivergenceAnalysis>();

  return Preserved;
}

bool ControlFlowConversionState::replaceReachableUses(Reachability &RC,
                                                      Instruction *from,
                                                      Value *to,
                                                      BasicBlock *src) {
  for (auto it = from->use_begin(); it != from->use_end();) {
    Use &U = *it++;
    Instruction *user = cast<Instruction>(U.getUser());

    if (user == to) {
      continue;
    }

    BasicBlock *blockUse = user->getParent();

    if (PHINode *PHI = dyn_cast<PHINode>(user)) {
      // Cannot replace a use in a phi node with another phi node in the same
      // block.
      if (blockUse == src) {
        if (isa<PHINode>(to)) {
          continue;
        }
      } else {
        // We must also check that 'src' can reach the incoming block to be
        // allowed to replace the incoming value.
        BasicBlock *incoming = PHI->getIncomingBlock(U);
        if (!RC.isReachable(src, incoming)) {
          continue;
        }
      }
    }

    if (auto toI = dyn_cast<Instruction>(to)) {
      if (toI->getParent() == blockUse) {
        for (Instruction &I : *src) {
          // If we found the user before `to`, then skip this user as it lives
          // before `to` in the same block.
          if (&I == user) {
            break;
          }
          if (&I == to) {
            LLVM_DEBUG(dbgs() << "Replace  " << *from << " with " << *to
                              << " in " << *user << "\n");
            U.set(to);
            break;
          }
        }
        // We've handled all possible cases if `to` lives in the same block as
        // `user`, so iterate over a new instruction.
        continue;
      }
    }

    // `to` is in a different block than `user` so just check for reachability
    // across BasicBlocks and not within them.
    if (RC.isReachable(src, blockUse)) {
      LLVM_DEBUG(dbgs() << "Replace  " << *from << " with " << *to << " in "
                        << *user << "\n");
      U.set(to);
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::convertToDataFlow() {
  DT = &AM.getResult<DominatorTreeAnalysis>(F);
  PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);
  LI = &AM.getResult<LoopAnalysis>(F);
  UVR = &AM.getResult<UniformValueAnalysis>(F);

  // Make sure every loop has an entry in the masks table before we start.
  for (auto *L : *LI) {
    LoopMasks[L];
  }

  if (!VU.choices().linearizeBOSCC()) {
    ROSCCGadget ROSCC(*this);
    ROSCC.run(F);
  }

  RC = std::make_unique<Reachability>(*DT, *PDT, *LI);

  // We do this after ROSCC, because it may have modified the CFG.
  DR = &AM.getResult<DivergenceAnalysis>(F);

  if (VU.choices().linearizeBOSCC()) {
    BOSCC = std::make_unique<BOSCCGadget>(*this);
    if (!BOSCC->duplicateUniformRegions()) {
      emitVeczRemarkMissed(&F, VU.scalarFunction(),
                           "Could not duplicate uniform regions for");
      return false;
    }
  }

  // Reserve space for the masks table and default-construct all entries, to
  // avoid re-hashing/element relocation on access.
  MaskInfos.reserve(F.size());
  for (auto &BB : F) {
    MaskInfos[&BB];
  }

  if (!generateMasks()) {
    emitVeczRemarkMissed(&F, VU.scalarFunction(),
                         "Could not generate masks for");
    return false;
  }
  if (auto err = applyMasks()) {
    emitVeczRemarkMissed(&F, VU.scalarFunction(), "Could not apply masks for",
                         llvm::toString(std::move(err)));
    return false;
  }

  if (!partiallyLinearizeCFG()) {
    emitVeczRemarkMissed(&F, VU.scalarFunction(),
                         "Could not partially linearize the CFG for");
    return false;
  }

  return true;
}

bool ControlFlowConversionState::Impl::generateMasks() {
  LLVM_DEBUG(dbgs() << "MASKS GENERATION\n");

  RC->update(F);

  VECZ_FAIL_IF(!createMasks(*functionExitBlock));

  if (BOSCC) {
    // The BOSCC entry blocks that have not been duplicated need exit masks
    // towards uniform blocks.
    SmallVector<BasicBlock *, 16> entryBlocks;
    BOSCC->getUnduplicatedEntryBlocks(entryBlocks);
    for (auto *const entry : entryBlocks) {
      VECZ_FAIL_IF(!createExitMasks(*entry, true));
    }

    // Link the masks of the predicated regions to the uniform regions.
    VECZ_FAIL_IF(!BOSCC->linkMasks());
  }

  for (auto *const LTag : DR->getLoopOrdering()) {
    VECZ_FAIL_IF(!createLoopExitMasks(*LTag));
  }

  return true;
}

bool ControlFlowConversionState::Impl::createMasks(BasicBlock &BB) {
  // If we have already set the mask for this block, don't do it again.
  // Uniform blocks are handled separately because of their lack of context.
  if (MaskInfos[&BB].entryMask) {
    return true;
  }

  auto *const LTag = DR->getTag(&BB).loop;
  auto *const header = LTag ? LTag->header : nullptr;
  // If BB is a header, we will need the mask from its preheader.
  // KLOCWORK "NPD.CHECK.MIGHT" possible false positive
  // LTag is only dereferenced if it's not nullptr, but Klocwork doesn't follow
  // the logic.
  if (header == &BB) {
    BasicBlock *preheader = LTag->preheader;
    VECZ_FAIL_IF(!createMasks(*preheader));
  } else {
    // Otherwise we will need the mask from every incoming edge.
    for (BasicBlock *pred : predecessors(&BB)) {
      VECZ_FAIL_IF(!createMasks(*pred));
    }
  }

  VECZ_FAIL_IF(!createEntryMasks(BB));
  VECZ_FAIL_IF(!createExitMasks(BB));

  // If the block is a loop header, its entry mask is a phi function with
  // incoming values from the preheader and:
  //  - the latch for divergent loops,
  //  - nothing else for uniform loops (because if we enter an uniform loop,
  //    all instance that were active upon entry remain active upon exit).
  if (header == &BB) {
    BasicBlock *latch = LTag->latch;
    VECZ_FAIL_IF(!createMasks(*latch));

    if (LTag->isLoopDivergent()) {
      auto *const entryMask = MaskInfos[&BB].entryMask;
      assert(isa<PHINode>(entryMask) &&
             "Divergent Loop entry mask must be a PHI Node!");
      PHINode *phi = cast<PHINode>(entryMask);
      // If header has two incoming values, we have already processed it.
      if (phi->getNumIncomingValues() != 2) {
        Value *latchMask = MaskInfos[latch].exitMasks[header];
        phi->addIncoming(latchMask, latch);

        LLVM_DEBUG(dbgs() << "Divergent loop header " << header->getName()
                          << ": entry mask: " << *phi << "\n");
      }
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::createEntryMasks(BasicBlock &BB) {
  auto &maskInfo = MaskInfos[&BB];
  if (maskInfo.entryMask) {
    return true;
  }

  Type *maskTy = Type::getInt1Ty(BB.getContext());

  // If the block is by_all (i.e. executed by all lanes), it will always be
  // executed on active masks,
  // Similarly, if the block is uniform, its mask is true by definition.
  if (DR->isByAll(BB) || DR->isUniform(BB)) {
    maskInfo.entryMask = copyEntryMask(getDefaultValue(maskTy, 1), BB);
    LLVM_DEBUG(dbgs() << BB.getName() << ": entry mask: " << *maskInfo.entryMask
                      << "\n");
    return true;
  }

  // If the block has only one predecessor, set its entry mask to be its
  // predecessor's exit mask.
  const unsigned numPreds = std::distance(pred_begin(&BB), pred_end(&BB));
  if (numPreds == 1) {
    BasicBlock *pred = *pred_begin(&BB);
    maskInfo.entryMask = copyEntryMask(MaskInfos[pred].exitMasks[&BB], BB);
    LLVM_DEBUG(dbgs() << BB.getName()
                      << ": entry mask: its single predecessor exit mask "
                      << *maskInfo.entryMask << "\n");
    return true;
  }

  // If the block is a loop header, its mask is a phi function with incoming
  // values from the preheader and:
  //  - the latch for divergent loops,
  //  - nothing else for uniform loops (because if we enter a uniform loop,
  //    all instance that were active upon entry remain active upon exit).
  //
  // Here we only store the preheader's exit block as we handle the latch
  // in case the loop is divergent in the caller function.
  const auto *const LTag = DR->getTag(&BB).loop;
  if (LTag && LTag->header == &BB) {
    BasicBlock *preheader = LTag->preheader;
    VECZ_ERROR_IF(!preheader, "BasicBlock tag is not defined");

    if (LTag->isLoopDivergent()) {
      PHINode *PHI = PHINode::Create(maskTy, 2, BB.getName() + ".entry_mask");
      multi_llvm::insertBefore(PHI, BB.begin());
      PHI->addIncoming(MaskInfos[preheader].exitMasks[&BB], preheader);
      maskInfo.entryMask = PHI;
      LLVM_DEBUG(dbgs() << "Loop divergent loop header " << BB.getName()
                        << ": entry mask: " << *maskInfo.entryMask << "\n");

    } else {
      maskInfo.entryMask =
          copyEntryMask(MaskInfos[preheader].exitMasks[&BB], BB);
      LLVM_DEBUG(dbgs() << "Uniform loop header " << BB.getName()
                        << ": entry mask: " << *maskInfo.entryMask << "\n");
    }
    return true;
  }

  // If the dominator of this block is also post-dominated by this block,
  // then if one is executed, the other must be also. So copy the mask.
  auto *IDom = DT->getNode(&BB)->getIDom();
  while (IDom) {
    BasicBlock *DomBB = IDom->getBlock();
    if (DR->getTag(DomBB).loop == LTag && PDT->dominates(&BB, DomBB)) {
      maskInfo.entryMask = copyEntryMask(MaskInfos[DomBB].entryMask, BB);
      LLVM_DEBUG(dbgs() << "Copied-via-domination " << BB.getName()
                        << ": entry mask: " << *maskInfo.entryMask << "\n");
      return true;
    }
    IDom = IDom->getIDom();
  }

  // In any other case, its mask is the disjunction of every incoming edge.
  // The union of every predecessor if it is a join point of a varying branch.
  if (DR->isBlend(BB)) {
    for (auto it = pred_begin(&BB); it != pred_end(&BB); ++it) {
      if (it == pred_begin(&BB)) {
        maskInfo.entryMask = copyEntryMask(MaskInfos[*it].exitMasks[&BB], BB);
        LLVM_DEBUG(dbgs() << "Blend block " << BB.getName()
                          << ": entry mask: " << *maskInfo.entryMask << "\n");
      } else {
        Instruction *insertBefore =
            cast<Instruction>(maskInfo.entryMask)->getNextNode();
        maskInfo.entryMask = BinaryOperator::CreateOr(
            maskInfo.entryMask, MaskInfos[*it].exitMasks[&BB],
            BB.getName() + ".entry_mask", insertBefore);

        LLVM_DEBUG(dbgs() << "Blend block " << BB.getName()
                          << ": entry mask: " << *maskInfo.entryMask << "\n");
      }
    }
  } else {
    // A phi function of the predecessors otherwise.
    PHINode *PHI =
        PHINode::Create(maskTy, numPreds, BB.getName() + ".entry_mask");
    multi_llvm::insertBefore(PHI, BB.begin());
    for (auto it = pred_begin(&BB); it != pred_end(&BB); ++it) {
      PHI->addIncoming(MaskInfos[*it].exitMasks[&BB], *it);
    }
    maskInfo.entryMask = PHI;
    LLVM_DEBUG(dbgs() << BB.getName() << ": entry mask: " << *maskInfo.entryMask
                      << "\n");
  }

  return true;
}

bool ControlFlowConversionState::Impl::createExitMasks(BasicBlock &BB,
                                                       bool isBOSCCEntry) {
  assert((!isBOSCCEntry || BOSCC) &&
         "Creating BOSCC Exit Masks when BOSCC object does not exist!");

  auto &maskInfo = MaskInfos[&BB];

  // If BB is a BOSCC entry, we want to compute the uniform exit masks for
  // this block.
  if (!isBOSCCEntry && !maskInfo.exitMasks.empty()) {
    return true;
  }

  const unsigned numSucc = std::distance(succ_begin(&BB), succ_end(&BB));

  // If BB has no successor, there is obviously nothing to do.
  if (numSucc == 0) {
    return true;
  }

  // If BB has only one successor, then the exit mask is the entry mask of BB.
  if (numSucc == 1) {
    BasicBlock *succ = *succ_begin(&BB);
    maskInfo.exitMasks[succ] =
        copyExitMask(maskInfo.entryMask, succ->getName(), BB);
    LLVM_DEBUG(dbgs() << BB.getName() << ": exit mask to single successor "
                      << succ->getName() << ": " << *maskInfo.entryMask
                      << "\n");
    return true;
  }

  const bool isVarying = DR->getTag(&BB).hasVaryingBranch();

  // If BB has more than 1 successor, the exit mask of each successor is the
  // conjunction of the entry mask of BB and the condition to jump to the
  // successor.
  auto *T = BB.getTerminator();
  IRBuilder<> B(T);

  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    BasicBlock *trueBB = BI->getSuccessor(0);
    BasicBlock *falseBB = BI->getSuccessor(1);
    assert(trueBB && "Could not get successor 0 of branch");
    assert(falseBB && "Could not get successor 1 of branch");

    if (isBOSCCEntry) {
      if (BasicBlock *trueBBUniform = BOSCC->getBlock(trueBB)) {
        trueBB = trueBBUniform;
      }
      if (BasicBlock *falseBBUniform = BOSCC->getBlock(falseBB)) {
        falseBB = falseBBUniform;
      }
    }

    Value *cond = BI->getCondition();
    if (isVarying) {
      maskInfo.exitMasks[trueBB] = B.CreateAnd(
          maskInfo.entryMask, cond, trueBB->getName() + ".exit_mask");

      // For the false edge, we have to negate the condition.
      Value *falseCond = B.CreateNot(cond, cond->getName() + ".not");
      maskInfo.exitMasks[falseBB] = B.CreateAnd(
          maskInfo.entryMask, falseCond, falseBB->getName() + ".exit_mask");

      LLVM_DEBUG(dbgs() << BB.getName() << ": varying exit mask to "
                        << trueBB->getName() << ": "
                        << *maskInfo.exitMasks[trueBB] << "\n");
      LLVM_DEBUG(dbgs() << BB.getName() << ": varying exit mask to "
                        << falseBB->getName() << ": "
                        << *maskInfo.exitMasks[falseBB] << "\n");
    } else {
      maskInfo.exitMasks[trueBB] = B.CreateSelect(
          cond, maskInfo.entryMask, getDefaultValue(cond->getType()),
          trueBB->getName() + ".exit_mask");
      maskInfo.exitMasks[falseBB] =
          B.CreateSelect(cond, getDefaultValue(cond->getType()),
                         maskInfo.entryMask, falseBB->getName() + ".exit_mask");

      LLVM_DEBUG(dbgs() << BB.getName() << ": uniform exit mask to "
                        << trueBB->getName() << ": "
                        << *maskInfo.exitMasks[trueBB] << "\n");
      LLVM_DEBUG(dbgs() << BB.getName() << ": uniform exit mask to "
                        << falseBB->getName() << ": "
                        << *maskInfo.exitMasks[falseBB] << "\n");
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    Value *cond = SI->getCondition();
    BasicBlock *defaultDest = SI->getDefaultDest();

    if (isBOSCCEntry) {
      if (BasicBlock *defaultDestUniform = BOSCC->getBlock(defaultDest)) {
        defaultDest = defaultDestUniform;
      }
    }

    // The default condition is the negation of the disjunction of every case
    // condition, so that if no case has its condition true, then we can choose
    // default.
    Value *caseConds = nullptr;
    for (auto c : SI->cases()) {
      Value *caseCond = B.CreateICmpEQ(cond, c.getCaseValue());
      caseConds = !caseConds ? caseCond : B.CreateOr(caseConds, caseCond);
      BasicBlock *caseBlock = c.getCaseSuccessor();
      if (isBOSCCEntry) {
        if (BasicBlock *caseBlockUniform = BOSCC->getBlock(caseBlock)) {
          caseBlock = caseBlockUniform;
        }
      }

      if (isVarying) {
        maskInfo.exitMasks[caseBlock] = B.CreateAnd(
            maskInfo.entryMask, caseCond, caseBlock->getName() + ".exit_mask");
        LLVM_DEBUG(dbgs() << BB.getName() << ": varying exit mask to "
                          << caseBlock->getName() << ": "
                          << *maskInfo.exitMasks[caseBlock] << "\n");
      } else {
        maskInfo.exitMasks[caseBlock] = B.CreateSelect(
            caseCond, maskInfo.entryMask, getDefaultValue(caseCond->getType()),
            caseBlock->getName() + ".exit_mask");
        LLVM_DEBUG(dbgs() << BB.getName() << ": uniform exit mask to "
                          << caseBlock->getName() << ": "
                          << *maskInfo.exitMasks[caseBlock] << "\n");
      }
    }

    VECZ_ERROR_IF(!caseConds, "No switch condition was found");

    Value *negCond = B.CreateNot(caseConds, caseConds->getName() + ".not");
    if (isVarying) {
      maskInfo.exitMasks[defaultDest] = B.CreateAnd(
          negCond, maskInfo.entryMask, defaultDest->getName() + ".exit_mask");
      LLVM_DEBUG(dbgs() << BB.getName() << ": varying exit mask to "
                        << defaultDest->getName() << ": "
                        << *maskInfo.exitMasks[defaultDest] << "\n");
    } else {
      maskInfo.exitMasks[defaultDest] = B.CreateSelect(
          negCond, maskInfo.entryMask, getDefaultValue(negCond->getType()),
          defaultDest->getName() + ".exit_mask");
      LLVM_DEBUG(dbgs() << BB.getName() << ": uniform exit mask to "
                        << defaultDest->getName() << ": "
                        << *maskInfo.exitMasks[defaultDest] << "\n");
    }
  } else {
    // We should not have a case where we don't have a BranchInst nor a
    // SwitchInst but more than 1 successors.
    return false;
  }

  return true;
}

bool ControlFlowConversionState::Impl::createLoopExitMasks(LoopTag &LTag) {
  auto &LMask = LoopMasks[LTag.loop];
  // If the Loop already has a CombinedExitMasks we have already processed it.
  if (LMask.combinedDivergentExitMask) {
    return true;
  }

  Type *maskTy = Type::getInt1Ty(F.getContext());
  SmallVector<Loop::Edge, 1> exitEdges;
  LTag.loop->getExitEdges(exitEdges);
  for (const Loop::Edge &EE : exitEdges) {
    const auto *const exitingBlock = EE.first;
    const auto *const exitBlock = EE.second;
    // Divergent loop need to keep track of which instance left at which exit.
    if (LTag.isLoopDivergent() && DR->isDivergent(*exitBlock)) {
      // The value of the exit mask of a divergent loop is a phi function
      // between the mask update and the loop exit mask phi.
      auto *const exitMask =
          PHINode::Create(maskTy, 2, exitBlock->getName() + ".loop_exit_mask");
      multi_llvm::insertBefore(exitMask,
                               multi_llvm::getFirstNonPHIIt(LTag.header));
      LMask.persistedDivergentExitMasks[exitingBlock] = exitMask;
      if (BOSCC) {
        BOSCC->createReference(exitMask, getDefaultValue(maskTy));
      }
    }
  }

  for (Loop *L : LTag.loop->getSubLoops()) {
    VECZ_FAIL_IF(!createLoopExitMasks(DR->getTag(L)));
  }

  // If the loop is uniform, all instances that enter the loop will leave it
  // together.
  if (!LTag.isLoopDivergent()) {
    return true;
  }

  // Check if the exit edge leaves multiple loops, in which case we return the
  // next inner loop left by it.
  auto nextInnerLoopLeft = [this, &LTag](BasicBlock *exitingBlock,
                                         BasicBlock *exitBlock) -> Loop * {
    Loop *innerLoop = nullptr;
    Loop *loop = DR->getTag(exitingBlock).loop->loop;
    // Iterate until we reach the current loop.
    while (loop && loop != LTag.loop) {
      // If this is an exit edge.
      if (loop->contains(exitingBlock) && !loop->contains(exitBlock)) {
        innerLoop = loop;
      }

      loop = loop->getParentLoop();
    }

    return innerLoop;
  };

  for (const Loop::Edge &EE : exitEdges) {
    BasicBlock *exitingBlock = const_cast<BasicBlock *>(EE.first);
    BasicBlock *exitBlock = const_cast<BasicBlock *>(EE.second);

    if (DR->isDivergent(*exitBlock)) {
      PHINode *REM = LMask.persistedDivergentExitMasks[exitingBlock];
      REM->addIncoming(getDefaultValue(REM->getType()), LTag.preheader);

      const auto *const exitingLTag = DR->getTag(exitingBlock).loop;
      VECZ_ERROR_IF(!exitingLTag, "Loop tag is not defined");

      // By default, the second operand of the mask update is the exit
      // condition.
      auto &exitMasks = MaskInfos[exitingBlock].exitMasks;
      Value *maskUpdateOperand = exitMasks[exitBlock];

      // If the exit leaves multiple loops and the current loop is not the
      // innermost left by this exit, set the update mask to be a disjunction
      // with the exit mask and the accumulated update mask from the next inner
      // loop left by this exit.
      if (exitingLTag->loop != LTag.loop) {
        if (Loop *nestedLoop = nextInnerLoopLeft(exitingBlock, exitBlock)) {
          maskUpdateOperand =
              LoopMasks[nestedLoop]
                  .updatedPersistedDivergentExitMasks[exitingBlock];
        }
      }

      BinaryOperator *maskUpdate = BinaryOperator::CreateOr(
          REM, maskUpdateOperand,
          exitBlock->getName() + ".loop_exit_mask.update",
          exitingBlock->getTerminator());

      LMask.updatedPersistedDivergentExitMasks[exitingBlock] = maskUpdate;

      if (BOSCC) {
        // The uniform version of divergent loop exit masks is the edge's exit
        // mask.
        BOSCC->addReference(maskUpdate, exitMasks[exitBlock]);
      }

      // If this is the outermost loop left by this exit, update the exit
      // mask.
      if (DR->getTag(exitBlock).outermostExitedLoop == &LTag) {
        VECZ_ERROR_IF(!isa<Instruction>(exitMasks[exitBlock]),
                      "Trying to replace uses of a value");
        VECZ_FAIL_IF(
            !replaceReachableUses(*RC, cast<Instruction>(exitMasks[exitBlock]),
                                  maskUpdate, exitBlock));

        exitMasks[exitBlock] = maskUpdate;
      }

      REM->addIncoming(maskUpdate, LTag.latch);

      LLVM_DEBUG(dbgs() << "Divergent loop " << LTag.loop->getName()
                        << ": divergent loop exit edges ["
                        << exitingBlock->getName() << " -> "
                        << exitBlock->getName() << "]: exit mask: " << *REM
                        << "\n");
      LLVM_DEBUG(dbgs() << "Divergent loop " << LTag.loop->getName()
                        << ": divergent loop exit edges ["
                        << exitingBlock->getName() << " -> "
                        << exitBlock->getName()
                        << "]: update exit mask: " << *maskUpdate << "\n");
    }
  }

  VECZ_FAIL_IF(!createCombinedLoopExitMask(LTag));

  return true;
}

bool ControlFlowConversionState::Impl::createCombinedLoopExitMask(
    LoopTag &LTag) {
  // Gather every information on every instance that left the loop in the
  // current iteration.
  SmallVector<Loop::Edge, 1> exitEdges;
  auto *const Loop = LTag.loop;
  Loop->getExitEdges(exitEdges);
  auto &LMask = LoopMasks[Loop];
  for (const Loop::Edge &EE : exitEdges) {
    BasicBlock *exitingBlock = const_cast<BasicBlock *>(EE.first);
    BasicBlock *exitBlock = const_cast<BasicBlock *>(EE.second);
    if (DR->isDivergent(*exitBlock)) {
      if (!LMask.combinedDivergentExitMask) {
        LMask.combinedDivergentExitMask = copyMask(
            LMask.updatedPersistedDivergentExitMasks[exitingBlock]->getOperand(
                1),
            Loop->getName() + ".combined_divergent_exit_mask",
            LTag.latch->getTerminator());

        LMask.persistedCombinedDivergentExitMask = copyMask(
            LMask.updatedPersistedDivergentExitMasks[exitingBlock],
            Loop->getName() + ".persisted_combined_divergent_exit_mask",
            LTag.latch->getTerminator());
      } else {
        LMask.combinedDivergentExitMask = BinaryOperator::CreateOr(
            LMask.combinedDivergentExitMask,
            LMask.updatedPersistedDivergentExitMasks[exitingBlock]->getOperand(
                1),
            Loop->getName() + ".combined_divergent_exit_mask",
            LTag.latch->getTerminator());

        LMask.persistedCombinedDivergentExitMask = BinaryOperator::CreateOr(
            LMask.persistedCombinedDivergentExitMask,
            LMask.updatedPersistedDivergentExitMasks[exitingBlock],
            Loop->getName() + ".persisted_combined_divergent_exit_mask",
            LTag.latch->getTerminator());
      }
    }
  }

  VECZ_ERROR_IF(!LMask.combinedDivergentExitMask ||
                    !LMask.persistedCombinedDivergentExitMask,
                "Divergent loop has no loop exit condition");

  LLVM_DEBUG(dbgs() << "Divergent loop " << LTag.loop->getName()
                    << ": current iteration combine divergent loop exit: "
                    << *LMask.combinedDivergentExitMask << "\n");
  LLVM_DEBUG(dbgs() << "Divergent loop " << LTag.loop->getName()
                    << ": whole loop combine divergent loop exit: "
                    << *LMask.persistedCombinedDivergentExitMask << "\n");

  return true;
}

Error ControlFlowConversionState::Impl::applyMasks() {
  for (auto &BB : F) {
    // Use masks with instructions that have side-effects.
    if (!DR->isUniform(BB) && !DR->isByAll(BB)) {
      auto *const entryMask = MaskInfos[&BB].entryMask;
      VECZ_ERROR_IF(!entryMask, "BasicBlock should have an entry mask");
      if (auto err = applyMask(BB, entryMask)) {
        return err;
      }
    }
  }
  return Error::success();
}

Error ControlFlowConversionState::Impl::applyMask(BasicBlock &BB, Value *mask) {
  // Packetization hasn't happened yet so this better be a scalar 1 bit int.
  assert(mask->getType()->isIntegerTy(1) && "CFG mask type should be int1");
  // Map the unmasked instruction with the masked one.
  DeletionMap toDelete;
  DenseMap<Value *, Value *> safeDivisors;

  for (Instruction &I : BB) {
    if (tryApplyMaskToBinOp(I, mask, toDelete, safeDivisors)) {
      continue;
    }
    std::optional<MemOp> memOp = MemOp::get(&I);
    // Turn loads and stores into masked loads and stores.
    if (memOp && (memOp->isLoad() || memOp->isStore())) {
      if (!tryApplyMaskToMemOp(*memOp, mask, toDelete)) {
        return makeStringError("Could not apply mask to MemOp", I);
      }
    } else if (auto *CI = dyn_cast<CallInst>(&I)) {
      // Turn calls into masked calls if possible.
      if (!applyMaskToCall(CI, mask, toDelete)) {
        return makeStringError("Could not apply mask to call instruction", I);
      }
    } else if (I.isAtomic() && !isa<FenceInst>(&I)) {
      // Turn atomics into calls to masked builtins if possible.
      if (!applyMaskToAtomic(I, mask, toDelete)) {
        return makeStringError("Could not apply mask to atomic instruction", I);
      }
    } else if (auto *branch = dyn_cast<BranchInst>(&I)) {
      // We have to be careful with infinite loops, because if they exist on a
      // divergent code path, they will always be entered and will hang the
      // kernel. Therefore, we replace the branch condition with the mask of
      // the preheader, to ensure they only loop if at least one lane is
      // actually executed.
      if (branch->isConditional()) {
        auto *const cond = dyn_cast<Constant>(branch->getCondition());
        if (cond && cond->isOneValue()) {
          auto *const loop = DR->getTag(&BB).loop;
          if (loop && loop->latch == &BB) {
            auto *const loopMask = MaskInfos[loop->preheader].entryMask;
            branch->setCondition(loopMask);
          }
        }
      }
    }
  }

  for (auto &pair : toDelete) {
    Instruction *unmasked = pair.first;
    Value *masked = pair.second;
    updateMaps(unmasked, masked);
    IRCleanup::deleteInstructionNow(unmasked);
  }

  return Error::success();
}

CallInst *ControlFlowConversionState::Impl::emitMaskedVersion(CallInst *CI,
                                                              Value *entryBit) {
  // Get the masked function
  Function *newFunction = Ctx.getOrCreateMaskedFunction(CI);
  VECZ_FAIL_IF(!newFunction);
  SmallVector<Value *, 8> fnArgs;
  for (unsigned i = 0; i < CI->arg_size(); ++i) {
    fnArgs.push_back(CI->getOperand(i));
  }
  fnArgs.push_back(entryBit);

  CallInst *newCI = CallInst::Create(newFunction, fnArgs, "", CI);
  newCI->setCallingConv(CI->getCallingConv());
  newCI->setAttributes(CI->getAttributes());

  return newCI;
}

bool ControlFlowConversionState::Impl::tryApplyMaskToBinOp(
    Instruction &I, Value *mask, DeletionMap &toDelete,
    DenseMap<Value *, Value *> &safeDivisors) {
  if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
    if (!VU.choices().isEnabled(VectorizationChoices::eDivisionExceptions)) {
      // we don't need to mask division operations if they don't trap
      return true;
    }
    // We might have to mask integer divides to avoid division errors.
    // NOTE we don't generate any specific error checks ourselves, on the
    // assumption that the incoming IR is already guarded against these,
    // so it is sufficient to use the mask generated from the CFG.
    bool isUnsigned = false;
    switch (binOp->getOpcode()) {
      case Instruction::UDiv:
      case Instruction::URem:
        isUnsigned = true;
        LLVM_FALLTHROUGH;
      case Instruction::SDiv:
      case Instruction::SRem: {
        auto *divisor = binOp->getOperand(1);
        // no need to mask divides by a constant..
        if (auto *C = dyn_cast<Constant>(divisor)) {
          if (C->isZeroValue()) {
            // Divides by constant zero can be a NOP since there is no
            // division by zero exception in OpenCL.
            auto *nop = binOp->getOperand(0);
            I.replaceAllUsesWith(nop);
            toDelete.emplace_back(&I, nop);
          }
        } else {
          auto &masked = safeDivisors[divisor];
          if (!masked) {
            // NOTE this function does not check for the pattern
            // "select (x eq 0) 1, x" or equivalent, so we might want to
            // write it ourselves, but Instruction Combining cleans it up.
            // NOTE that for a signed division, we also have to consider the
            // potential overflow situation, which is not so simple
            if (isUnsigned &&
                isKnownNonZero(divisor, F.getParent()->getDataLayout())) {
              // Static analysis concluded it can't be zero, so we don't need
              // to do anything.
              masked = divisor;
            } else {
              masked = SelectInst::Create(
                  mask, divisor, ConstantInt::get(divisor->getType(), 1),
                  divisor->getName() + ".masked", &I);
            }
          }

          if (masked != divisor) {
            binOp->setOperand(1, masked);
          }
        }
      } break;

      default:
        break;
    }
    return true;
  } else {
    return false;
  }
}

bool ControlFlowConversionState::Impl::tryApplyMaskToMemOp(
    MemOp &memOp, Value *mask, DeletionMap &toDelete) {
  VECZ_FAIL_IF(!memOp.isLoad() && !memOp.isStore());
  auto *I = memOp.getInstr();
  VECZ_FAIL_IF(!I);
  auto *dataVecTy = dyn_cast<FixedVectorType>(memOp.getDataType());
  const unsigned dataWidth = dataVecTy ? dataVecTy->getNumElements() : 1;
  Value *wideMask = mask;
  if (dataWidth > 1) {
    // If it's a vector mem-op it gets the same mask for every element
    IRBuilder<> B(I);
    wideMask = B.CreateVectorSplat(dataWidth, mask);
  }

  // Turn loads and stores into masked loads and stores.
  if (memOp.isLoadStoreInst()) {
    // Create a new mem-op the same as the original except for the addition
    // of the mask.
    Value *newVal = nullptr;
    if (memOp.isLoad()) {
      newVal = createMaskedLoad(
          Ctx, memOp.getDataType(), memOp.getPointerOperand(), wideMask,
          /*VL*/ nullptr, memOp.getAlignment(), I->getName(), I);
    } else {
      newVal = createMaskedStore(
          Ctx, memOp.getDataOperand(), memOp.getPointerOperand(), wideMask,
          /*VL*/ nullptr, memOp.getAlignment(), I->getName(), I);
    }

    VECZ_FAIL_IF(!newVal);
    if (!I->getType()->isVoidTy()) {
      I->replaceAllUsesWith(newVal);
    }
    toDelete.emplace_back(I, newVal);
    return true;
  }

  if (auto *opMask = memOp.getMaskOperand()) {
    memOp.setMaskOperand(
        BinaryOperator::CreateAnd(wideMask, opMask, "composite_mask", I));
    return true;
  }

  return false;
}

bool ControlFlowConversionState::Impl::applyMaskToCall(CallInst *CI,
                                                       Value *mask,
                                                       DeletionMap &toDelete) {
  LLVM_DEBUG(dbgs() << "vecz-cf: Now at CallInst " << *CI << "\n");
  // It might be that we need to mask the function call here because we
  // won't be able to packetize it later on.
  Function *callee = CI->getCalledFunction();
  if (!callee) {
    callee = dyn_cast<Function>(CI->getCalledOperand()->stripPointerCasts());
  }
  VECZ_FAIL_IF(!callee);  // TODO: CA-1505: Support indirect function calls.
  // Check to see if this is a function that we know we won't be able to
  // handle in any other way.
  VECZ_FAIL_IF(callee->cannotDuplicate());

  // Do not mess with internal builtins
  if (Ctx.isInternalBuiltin(callee)) {
    LLVM_DEBUG(dbgs() << "vecz-cf: Called function is an internal builtin\n");
    return true;
  }

  // Builtins without side effects do not need to be masked.
  const auto builtin = Ctx.builtins().analyzeBuiltin(*callee);
  const auto props = builtin.properties;
  if (props & compiler::utils::eBuiltinPropertyNoSideEffects) {
    LLVM_DEBUG(dbgs() << "vecz-cf: Called function is an pure builtin\n");
    return true;
  }
  if (props & compiler::utils::eBuiltinPropertyWorkItem) {
    LLVM_DEBUG(dbgs() << "vecz-cf: Called function is a workitem ID builtin\n");
    return true;
  }
  if (props & compiler::utils::eBuiltinPropertyExecutionFlow) {
    LLVM_DEBUG(
        dbgs() << "vecz-cf: Called function is an execution flow builtin\n");
    // Masking this kind of builtin (a barrier) is not valid.
    return false;
  }
  // Functions without side-effects do not need to be masked.
  if (callee->onlyReadsMemory() || callee->doesNotAccessMemory()) {
    LLVM_DEBUG(
        dbgs() << "vecz-cf: Called function does not have any side-effects\n");
    return true;
  }
  // We don't want to mask work-group collective builtins, because they are
  // barriers (see above). This should actually be a rare situation, as these
  // builtins are required to be uniform/convergent and so either all
  // work-items or no work-items should hit them. Most of the time, this
  // situation relies on the vectorizer failing to trace the branch flow and
  // failing to realize the conditions are in fact uniform.
  if (auto info = Ctx.builtins().isMuxGroupCollective(builtin.ID);
      info && info->isWorkGroupScope()) {
    LLVM_DEBUG(
        dbgs() << "vecz-cf: Called function is a work-group collective\n");
    return true;
  }

  // Create the new function and replace the old one with it
  CallInst *newCI = emitMaskedVersion(CI, mask);
  VECZ_FAIL_IF(!newCI);
  if (!CI->getType()->isVoidTy()) {
    CI->replaceAllUsesWith(newCI);
  }
  toDelete.emplace_back(CI, newCI);

  LLVM_DEBUG(dbgs() << "vecz-cf: Replaced " << *CI << "\n");
  LLVM_DEBUG(dbgs() << "          with " << *newCI << "\n");

  return true;
}

bool ControlFlowConversionState::Impl::applyMaskToAtomic(
    Instruction &I, Value *mask, DeletionMap &toDelete) {
  LLVM_DEBUG(dbgs() << "vecz-cf: Now at atomic inst " << I << "\n");

  SmallVector<Value *, 8> maskedFnArgs;
  VectorizationContext::MaskedAtomic MA;
  MA.VF = ElementCount::getFixed(1);
  MA.IsVectorPredicated = VU.choices().vectorPredication();

  if (auto *atomicI = dyn_cast<AtomicRMWInst>(&I)) {
    MA.Align = atomicI->getAlign();
    MA.BinOp = atomicI->getOperation();
    MA.IsVolatile = atomicI->isVolatile();
    MA.Ordering = atomicI->getOrdering();
    MA.SyncScope = atomicI->getSyncScopeID();
    MA.ValTy = atomicI->getType();
    MA.PointerTy = atomicI->getPointerOperand()->getType();

    // Set up the arguments to this function
    maskedFnArgs = {atomicI->getPointerOperand(), atomicI->getValOperand(),
                    mask};

  } else if (auto *cmpxchgI = dyn_cast<AtomicCmpXchgInst>(&I)) {
    MA.Align = cmpxchgI->getAlign();
    MA.BinOp = AtomicRMWInst::BAD_BINOP;
    MA.IsWeak = cmpxchgI->isWeak();
    MA.IsVolatile = cmpxchgI->isVolatile();
    MA.Ordering = cmpxchgI->getSuccessOrdering();
    MA.CmpXchgFailureOrdering = cmpxchgI->getFailureOrdering();
    MA.SyncScope = cmpxchgI->getSyncScopeID();
    MA.ValTy = cmpxchgI->getCompareOperand()->getType();
    MA.PointerTy = cmpxchgI->getPointerOperand()->getType();

    // Set up the arguments to this function
    maskedFnArgs = {cmpxchgI->getPointerOperand(),
                    cmpxchgI->getCompareOperand(), cmpxchgI->getNewValOperand(),
                    mask};
  } else {
    return false;
  }

  // Create the new function and replace the old one with it
  // Get the masked function
  Function *maskedAtomicFn = Ctx.getOrCreateMaskedAtomicFunction(
      MA, VU.choices(), ElementCount::getFixed(1));
  VECZ_FAIL_IF(!maskedAtomicFn);
  // We don't have a vector length just yet - pass in one as a dummy.
  if (MA.IsVectorPredicated) {
    maskedFnArgs.push_back(
        ConstantInt::get(IntegerType::getInt32Ty(I.getContext()), 1));
  }

  CallInst *maskedCI = CallInst::Create(maskedAtomicFn, maskedFnArgs, "", &I);
  VECZ_FAIL_IF(!maskedCI);

  I.replaceAllUsesWith(maskedCI);
  toDelete.emplace_back(&I, maskedCI);

  LLVM_DEBUG(dbgs() << "vecz-cf: Replaced " << I << "\n");
  LLVM_DEBUG(dbgs() << "          with " << *maskedCI << "\n");

  return true;
}

bool ControlFlowConversionState::Impl::partiallyLinearizeCFG() {
  // Two methods are possible to transform the divergent loops into uniform
  // ones:
  // 1) rewire the exit edges to the single latch, which means the loop live
  //    masks have to be updated at each exiting block.
  // 2) delete the divergent loop exit edges and update the loop live masks at
  //    the latch.
  //
  // The former means more overhead when a loop exit is reached because we
  // always have to update the masks, but it allows to retain the exiting
  // branches.
  // The latter means we only blend at the latch, thus less overhead at the
  // loop exits, but if we reach a divergent loop exit, and it happens that all
  // lanes have exited the loop, we still have to finish the iteration until we
  // reach the latch and exit the loop.
  //
  // We are currently using the latter.
  VECZ_FAIL_IF(!uniformizeDivergentLoops());

  // ... and actually rewire them.
  VECZ_FAIL_IF(!linearizeCFG());

  // Transform phi nodes into selects for blocks that got blended.
  VECZ_FAIL_IF(!generateSelects());

  // Connect BOSCC regions if it is activated.
  VECZ_FAIL_IF(BOSCC && !BOSCC->connectBOSCCRegions());

  // Repair the CFG because the rewiring broke it.
  VECZ_FAIL_IF(!repairSSA());

  // Now we create the opaque calls to builtins that compute the real branch
  // values. This must come before instruction simplification, otherwise LLVM
  // can fold branch predicates that appear unreachable now, but would later
  // become vector masks, thus mangling the control flow..
  VECZ_FAIL_IF(!createBranchReductions());

  // ... and now we can do instruction simplification on the masks and know they
  // won't be prematurely folded.
  VECZ_FAIL_IF(!simplifyMasks());

  // Finally, if we used BOSCC it might want to do some tidying up.
  VECZ_FAIL_IF(BOSCC && !BOSCC->cleanUp());

  return true;
}

bool ControlFlowConversionState::Impl::createBranchReductions() {
  // Try to retrieve the builtin if it already exists.
  const auto baseName =
      Twine(VectorizationContext::InternalBuiltinPrefix).concat("divergence");
  const StringRef nameAny = "_any";
  const StringRef nameAll = "_all";

  Type *boolTy = Type::getInt1Ty(F.getContext());
  FunctionType *FT = FunctionType::get(boolTy, {boolTy}, false);

  for (BasicBlock &BB : F) {
    const bool needsAllOfMask = DR->hasFlag(BB, eBlockNeedsAllOfMask);

    // If the block is uniform and is not a bossc indirection, all its lanes
    // are true or false, not both. Thus, we don't need to packetize the
    // condition.
    if (!needsAllOfMask && DR->isUniform(BB)) {
      continue;
    }

    auto *TI = BB.getTerminator();
    if (BranchInst *Branch = dyn_cast<BranchInst>(TI)) {
      if (Branch->isConditional()) {
        auto *cond = Branch->getCondition();
        if (isa<Constant>(cond)) {
          continue;
        }

        // On divergent paths, ensure that only active lanes contribute to a
        // branch condition; merge the branch condition with the active lane
        // mask. This ensures that disabled lanes don't spuriously contribute a
        // 'true' value into the reduced branch condition.
        // Note that the distinction between 'uniform' and 'divergent' isn't
        // 100% sufficient for our purposes here, because even uniform values
        // may read undefined/poison values when masked out.
        // Don't perform this on uniform loops as those may be unconditionally
        // entered even when no work-items are active. Masking the loop exit
        // with the entry mask would mean that the loop never exits.
        // FIXME: Is this missing incorrect branches in uniform blocks/loops?
        if (auto *LTag = DR->getTag(&BB).loop;
            DR->isDivergent(BB) && (!LTag || LTag->isLoopDivergent())) {
          if (!isBranchCondTrulyUniform(cond, *UVR)) {
            cond = BinaryOperator::Create(Instruction::BinaryOps::And, cond,
                                          MaskInfos[&BB].entryMask,
                                          cond->getName() + "_active", Branch);
          }
        }

        const auto &name = needsAllOfMask ? nameAll : nameAny;
        Function *const F = Ctx.getOrCreateInternalBuiltin(
            Twine(baseName).concat(name).str(), FT);
        VECZ_FAIL_IF(!F);

        auto *const newCall = CallInst::Create(
            F, {cond}, Twine(cond->getName()).concat(name), Branch);
        Branch->setCondition(newCall);
      }
    } else if (isa<SwitchInst>(TI) &&
               DR->hasFlag(BB, eBlockHasDivergentBranch)) {
      // Not sure what to actually do with switch instructions..
      return false;
    }
  }
  return true;
}

bool ControlFlowConversionState::Impl::uniformizeDivergentLoops() {
  LLVM_DEBUG(dbgs() << "CFC: UNIFORMIZE DIVERGENT LOOPS\n");

  // For every divergent loop of the function, we want to create a new exit edge
  // whose source is the latch of the loop. That exit is called "pure". The
  // target of this edge is a new divergent loop exit that will start a cascade
  // of if conditions to branch to the original loop exits. The divergent loop
  // exits will no longer be exits, while the optional loop exits will retain
  // their branch but they will be rewired to the pure exit.
  //
  // Given the following *divergent* loop:
  //
  //                           preheader
  //                               |
  //                             header <---------.
  //                              / \             |
  //                            ... ...           |
  //                            /     \           |
  //                     %exit2.o     ...         |
  //                     /            / \         |
  //                    %d     %exit1.o ...       |
  //                           /          \       |
  //                          %b          ...     |
  //                                      / \     |
  //                               %exit2.r ...   |
  //                               /          \   |
  //                              %c   %latch.r --'
  //                                   /
  //                            %exit1.r
  //                               |
  //                               %a
  //
  // with:
  // - %a, %b, %c, %d = a group of non specific basic blocks
  // - %exit*.*       = loop exits
  // - *.o            = optional blocks
  // - *.r            = divergent blocks
  // - %latch.r       = the latch of the loop. It is necessarily a divergent
  //                    block because the loop is divergent
  //
  // The following transformation is performed:
  //
  //                     preheader
  //                         |
  //                       header <---------.
  //                        / \             |
  //                      ... ...           |
  //                      /     \           |
  //        %exit2.split1.o     ...         |
  //        |                   / \         |
  //         \    %exit1.split1.o ...       |
  //          \   |                 \       |
  //           \   \                ...     |
  //            \   \                 \     |
  //             \   \                ...   |
  //              \   \                 \   |
  //               \   \         %latch.r --'
  //                \   \           |
  //                 `---`-> %loop.pure_exit
  //                               / |
  //                        %exit1.r %exit1.else.r
  //                        /             / |
  //                       %a      %exit2.r %exit2.else.r
  //                               /             / |
  //                              %c            /  |
  //                                           /   |
  //                             %exit1.split2.o   %exit1.else.o
  //                             /                      / |
  //                            %b        %exit2.split2.o %exit2.else.o
  //                                      /
  //                                     %d
  //
  // with:
  // - %exit*.split1.o = the first half of the original %exit*.o with only
  //                     phi nodes
  // - %exit*.split2.o = the second half of the original %exit*.o without the
  //                     phi nodes
  // - %loop.pure_exit = a new loop exit starting a cascade of ifs towards the
  //                     original loop exits
  // - %exit*.else.*   = a new block whose only purpose is to branch to other
  //                     blocks
  //
  // Each introduced conditional branch uses the entry mask of the exit block
  // as the condition.
  // Each introduced divergent conditional block is marked as Div causing, thus
  // linearizing them.
  // Each introduced optional conditional block is marked as divergent, thus
  // retaining the branches and branching to the true path only if any of the
  // lanes that executed the loop left through the exit the true path targets.
  //
  // The state of the loop after the transformation is invalid and relies on
  // the linearizer to correctly rewire the introduced blocks. The result of the
  // above transformed loop after linearization will be:
  //
  //                            preheader
  //                                |
  //                              header <---------.
  //                               / \             |
  //                             ... ...           |
  //                             /     \           |
  //               %exit2.split1.o     ...         |
  //                      |              \         |
  //                      |              ...       |
  //                      |                \       |
  //                      |                ...     |
  //                      |                / \     |
  //                      |  %exit1.split1.o ...   |
  //                       \        |          \   |
  //                        \       |   %latch.r --'
  //                         \      |      |
  //                          `---> %loop.pure_exit
  //                                       |
  //                                    %exit1.r
  //                                       |
  //                                       %a
  //                                       |
  //                                 %exit1.else.r
  //                                       |
  //                                    %exit2.r
  //                                       |
  //                                       %c
  //                                       |
  //                                 %exit2.else.r
  //                                      / |
  //                         %exit1.split.o %exit1.else.o
  //                         /                   / |
  //                        %b      %exit2.split.o %exit2.else.o
  //                                /                   ...
  //                               %d
  //
  // Note that only one branch introduced from an optional loop exit
  // ('%exit2.else.r' and '%exit1.else.o' in this example) can evaluate to
  // true because as soon as an optional loop exit is taken, all the active
  // lanes in the loop leave through it.
  // However, as many as all the branches introduced from divergent loop exits
  // may evaluate to true. The '...' at the end of the CFG will be replaced by
  // whatever would originally succeed the original divergent loop exits.
  bool modified = false;
  for (auto *const LTag : DR->getLoopOrdering()) {
    if (LTag->isLoopDivergent()) {
      Loop *L = LTag->loop;

      // Store the loop exit blocks and edges before doing any modification.
      SmallVector<BasicBlock *, 2> exitBlocks;
      SmallVector<Loop::Edge, 2> exitEdges;
      {
        L->getExitEdges(exitEdges);
        // 1) Retrieve the unique loop exit blocks.
        // 2) Remove any loop exit for which 'L' is not the outermost loop left.
        // 3) Sort the loop exit blocks.
        //
        // We can't use the `getUniqueExitBlocks' method because the loop may
        // not be in a canonical form because of BOSCC.
        if (BOSCC) {
          L->getExitBlocks(exitBlocks);
          SmallPtrSet<BasicBlock *, 1> _uniqueExitBlocks;
          for (auto it = exitBlocks.begin(); it != exitBlocks.end();) {
            if (!_uniqueExitBlocks.insert(*it).second) {
              it = exitBlocks.erase(it);
            } else {
              ++it;
            }
          }
        } else {
          L->getUniqueExitBlocks(exitBlocks);
        }
        // Only handle outermost loops left by the exits.
        exitBlocks.erase(
            std::remove_if(exitBlocks.begin(), exitBlocks.end(),
                           [this, LTag](BasicBlock *EB) {
                             return DR->getTag(EB).outermostExitedLoop != LTag;
                           }),
            exitBlocks.end());
        // Order the loop exit blocks such that:
        // - divergent loop exits come first
        // - smallest DCBI come first
        const auto middle = std::partition(
            exitBlocks.begin(), exitBlocks.end(),
            [this](BasicBlock *BB) { return DR->isDivergent(*BB); });
        std::sort(exitBlocks.begin(), middle,
                  [this](BasicBlock *LHS, BasicBlock *RHS) {
                    return DR->getTagIndex(LHS) < DR->getTagIndex(RHS);
                  });
        std::sort(middle, exitBlocks.end(),
                  [this](BasicBlock *LHS, BasicBlock *RHS) {
                    return DR->getTagIndex(LHS) < DR->getTagIndex(RHS);
                  });
      }

      if (exitBlocks.empty()) {
        LLVM_DEBUG(dbgs() << "Loop " << L->getName()
                          << " has no loop exits eligible for rewiring.\n");
        continue;
      }

      VECZ_FAIL_IF(!computeDivergentLoopPureExit(*LTag));
      VECZ_FAIL_IF(!rewireDivergentLoopExitBlocks(*LTag, exitBlocks));

      VECZ_FAIL_IF(!generateDivergentLoopResults(*LTag));
      VECZ_FAIL_IF(!blendDivergentLoopLiveValues(*LTag, exitBlocks));
      VECZ_FAIL_IF(!blendDivergentLoopExitMasks(*LTag, exitEdges, exitBlocks));

      modified = true;
    }
  }

  // We have modified the divergent loops into uniform ones, thus changing the
  // dominance-compact block ordering. We need to recompute it.
  if (modified) {
    DT->recalculate(F);
    PDT->recalculate(F);
    // And make sure we correctly updated the DomTrees.
    VECZ_ERROR_IF(!DT->verify(), "DominatorTree incorrectly updated");
    VECZ_ERROR_IF(!PDT->verify(), "PostDominatorTree incorrectly updated");
    VECZ_FAIL_IF(!computeBlockOrdering());

    RC->clear();
  }

  return true;
}

bool ControlFlowConversionState::Impl::computeDivergentLoopPureExit(
    LoopTag &LTag) {
  LLVM_DEBUG(dbgs() << "CFC: COMPUTE PURE EXIT FOR LOOP "
                    << LTag.loop->getName() << "\n");

  auto *const latchBB = LTag.latch;
  BasicBlock *pureExit =
      BasicBlock::Create(F.getContext(), LTag.loop->getName() + ".pure_exit",
                         &F, latchBB->getNextNode());
  BasicBlockTag &pureExitTag = DR->getOrCreateTag(pureExit);

  // Set the tags.
  auto &LMask = LoopMasks[LTag.loop];
  MaskInfos[pureExit].entryMask = LMask.persistedCombinedDivergentExitMask;
  pureExitTag.outermostExitedLoop = &LTag;

  auto *const preheaderLoopTag = DR->getTag(LTag.preheader).loop;
  if (preheaderLoopTag) {
    pureExitTag.loop = preheaderLoopTag;
    preheaderLoopTag->loop->addBasicBlockToLoop(pureExit, *LI);
  }
  DR->setFlag(*pureExit,
              static_cast<BlockDivergenceFlag>(
                  BlockDivergenceFlag::eBlockIsVirtualDivergentLoopExit |
                  BlockDivergenceFlag::eBlockHasDivergentBranch |
                  BlockDivergenceFlag::eBlockIsDivergent));

  LTag.pureExit = pureExit;

  LLVM_DEBUG(dbgs() << "Pure exit: " << pureExit->getName() << "\n");

  if (BOSCC) {
    BOSCC->addInRegions(pureExit, latchBB);
  }

  auto *latchT = latchBB->getTerminator();
#ifndef ALL_OF_DIVERGENT_LOOP_LATCH
  Value *cond = MaskInfos[latchBB].exitMasks[LTag.header];
  auto *newT = BranchInst::Create(LTag.header, pureExit, cond, latchBB);
#else
  // Exit the loop through the single divergent loop exit only if all instances
  // that entered the loop left it.
  ICmpInst *cond = new ICmpInst(
      latchT, CmpInst::ICMP_EQ, LMask.persistedCombinedDivergentExitMask,
      MaskInfos[LTag.preheader].exitMasks[LTag.header]);
  auto *newT = BranchInst::Create(pureExit, LTag.header, cond, latchBB);
  DR->setFlag(*latchBB, eBlockNeedsAllOfMask);
#endif

  updateMaps(latchT, newT);

  IRCleanup::deleteInstructionNow(latchT);

  MaskInfos[latchBB].exitMasks[pureExit] =
      LMask.persistedCombinedDivergentExitMask;

  return true;
}

bool ControlFlowConversionState::Impl::rewireDivergentLoopExitBlocks(
    LoopTag &LTag, const SmallVectorImpl<BasicBlock *> &exitBlocks) {
  LLVM_DEBUG(dbgs() << "CFC: REWIRE EXIT BLOCKS FOR LOOP "
                    << LTag.loop->getName() << "\n");

  auto removeSuccessor = [this](Instruction *T, unsigned succIdx) {
    switch (T->getOpcode()) {
      default:
        // Any other kind of Terminator cannot be handled and until
        // proven otherwise, should not.
        break;
      case Instruction::Br: {
        const unsigned keepIdx = succIdx == 0 ? 1 : 0;
        auto *newT = BranchInst::Create(T->getSuccessor(keepIdx), T);

        updateMaps(T, newT);

        IRCleanup::deleteInstructionNow(T);
        break;
      }
      case Instruction::Switch: {
        SwitchInst *SI = cast<SwitchInst>(T);
        if (succIdx == 0) {
          SI->setDefaultDest(SI->getSuccessor(1));
          SI->removeCase(SI->case_begin());
        } else {
          SI->removeCase(std::next(SI->case_begin(), succIdx - 1));
        }
        break;
      }
      case Instruction::IndirectBr: {
        IndirectBrInst *IBI = cast<IndirectBrInst>(T);
        IBI->removeDestination(succIdx);
        break;
      }
    }
  };

  // 'divergentLE' represents the current virtual divergent loop exit that a
  // loop exit needs to be rewired to/from.
  BasicBlock *divergentLE = LTag.pureExit;
  for (unsigned idx = 0; idx < exitBlocks.size(); ++idx) {
    BasicBlock *EB = exitBlocks[idx];

    // The target of 'divergentLE'.
    BasicBlock *target = nullptr;

    // If 'EB' is optional, we split it at the terminator so that the exiting
    // block keeps its edge towards it. The second half of 'EB' will be targeted
    // by the cascade if.
    if (DR->isOptional(*EB)) {
      LLVM_DEBUG(dbgs() << "Optional loop exit " << EB->getName() << ":\n");

      target =
          EB->splitBasicBlock(EB->getTerminator(), EB->getName() + ".split");
      auto &targetTag = DR->getOrCreateTag(target);

      LLVM_DEBUG(dbgs() << "\tSplit " << EB->getName() << " into "
                        << target->getName() << "\n");

      // Set the tags.
      // We have to be very careful copying a value from one key to another, in
      // case one key did not exist, and constructing it caused rehashing.
      {
        auto EBmasks = MaskInfos[EB];
        MaskInfos[target] = std::move(EBmasks);
      }

      auto *const EBLoopTag = DR->getTag(EB).loop;
      if (EBLoopTag) {
        targetTag.loop = EBLoopTag;
        EBLoopTag->loop->addBasicBlockToLoop(target, *LI);
      }

      // If 'EB' is the preheader of a loop then 'target' takes its place.
      for (auto *const ordered : DR->getLoopOrdering()) {
        if (ordered->preheader == EB) {
          LLVM_DEBUG(dbgs()
                     << "\t" << target->getName() << " is now the preheader of "
                     << ordered->loop->getName() << "\n");
          ordered->preheader = target;
        }
      }

      if (BOSCC) {
        BOSCC->addReference(target, EB);
        BOSCC->addInRegions(target, EB);
      }
      DR->setFlag(*target, DR->getFlag(*EB));

      // Rewire 'EB' to the pure exit.
      auto *const pureExit = LTag.pureExit;
      EB->getTerminator()->setSuccessor(0, pureExit);

      LLVM_DEBUG(dbgs() << "\t" << EB->getName() << " now targets "
                        << pureExit->getName() << "\n");

      // Retain branch for optional loop exits.
      DR->clearFlag(*divergentLE,
                    BlockDivergenceFlag::eBlockHasDivergentBranch);
      // Set all-of mask because the first successor of 'divergentLE' is taken
      // if no one existed from the optional loop exit.
      DR->setFlag(*divergentLE, eBlockNeedsAllOfMask);

      // 'EB' now has only one single exit edge.
      auto &EBmasks = MaskInfos[EB];
      EBmasks.exitMasks[pureExit] = EBmasks.entryMask;
    } else {
      LLVM_DEBUG(dbgs() << "Divergent loop exit " << EB->getName() << ":\n");

      // Otherwise, the edge exiting-block-to-divergent-exit-block is removed ..
      {
        SmallPtrSet<BasicBlock *, 1> predsToRemove;
        for (BasicBlock *pred : predecessors(EB)) {
          const auto *const predLTag = DR->getTag(pred).loop;
          // All predecessors of the divergent loop exit that belong in a loop
          // contained in the outermost loop left by that exit need their
          // edge removed.
          if (predLTag && LTag.loop->contains(predLTag->loop)) {
            predsToRemove.insert(pred);
          }
        }
        for (BasicBlock *pred : predsToRemove) {
          auto *predT = pred->getTerminator();
          for (unsigned succIdx = 0; succIdx < predT->getNumSuccessors();
               ++succIdx) {
            if (predT->getSuccessor(succIdx) == EB) {
              removeSuccessor(predT, succIdx);
              LLVM_DEBUG(dbgs() << "\tRemove predecessor: " << pred->getName()
                                << "\n");

              break;
            }
          }
        }
        PHINode *PHI = nullptr;
        while ((PHI = dyn_cast<PHINode>(&EB->front()))) {
          VECZ_FAIL_IF(!generateSelectFromPHI(PHI, EB));
        }
      }

      // ... and the exit block gets targeted by the current divergent loop
      // exit.
      target = EB;
    }

    VECZ_ERROR_IF(!target, "No target was found");

    // If we are processing the last exit block, and it happens to be divergent
    // there is no optional exit loop it can branch to, so create an
    // unconditional branch.
    if ((idx + 1 == exitBlocks.size()) && DR->isDivergent(*target)) {
      BranchInst::Create(target, divergentLE);
      auto &maskInfo = MaskInfos[divergentLE];
      maskInfo.exitMasks[target] = maskInfo.entryMask;

      LLVM_DEBUG(dbgs() << "\tVirtual Divergent Loop Exit "
                        << divergentLE->getName()
                        << ":\n\t\tSuccessor 0: " << target->getName() << "\n");
    } else {
      // The DCBI ordering sets the right sibling to be of an index less than
      // the left sibling if they are on the same level of dominance. For that
      // reason, we want to set the original loop exit as the right sibling so
      // that the latter gets processed first while linearizing, and branches
      // to the left sibling. We thus have to negate the condition to do so.
      //
      // The said condition is the entry mask of the exit block, i.e. whether
      // any exiting block left through it.
      auto &targetMasks = MaskInfos[target];
      Instruction *cond = cast<Instruction>(targetMasks.entryMask);
      // If that entry mask is defined in the loop (if the exit block has only
      // one predecessor), then we can directly use that mask as the condition.
      // Otherwise, we must move the latter in the pure exit so that
      // 'divergentLE' can refer to it.
      if (cond->getParent() == target) {
        if (PHINode *PHI = dyn_cast<PHINode>(cond)) {
          VECZ_FAIL_IF(!generateSelectFromPHI(PHI, target));
          cond = cast<Instruction>(targetMasks.entryMask);
        }
        std::queue<Instruction *> toMove;
        toMove.push(cond);
        // Make sure to move all the operands of the condition that are in its
        // definition block.
        while (!toMove.empty()) {
          Instruction *move = toMove.front();
          toMove.pop();
          move->moveBefore(*LTag.pureExit, LTag.pureExit->begin());
          for (Value *op : move->operands()) {
            if (Instruction *opI = dyn_cast<Instruction>(op)) {
              if (opI->getParent() == target) {
                toMove.push(opI);
              }
            }
          }
        }
      }

      auto *negCond = BinaryOperator::CreateNot(cond, cond->getName() + ".not",
                                                divergentLE);
      BasicBlock *newDivergentLE = BasicBlock::Create(
          F.getContext(), EB->getName() + ".else", &F, EB->getNextNode());
      BranchInst::Create(newDivergentLE, target, negCond, divergentLE);

      // The divergentLE block "ought" to exist in the masks map already, but
      // it is safer to take a local copy and retire `targetMasks`.
      auto *const targetEntryMask = targetMasks.entryMask;

      // No use of `targetMasks` after this line
      auto &divgLEMask = MaskInfos[divergentLE];
      divgLEMask.exitMasks[target] = targetEntryMask;
      divgLEMask.exitMasks[newDivergentLE] = negCond;

      LLVM_DEBUG(dbgs() << "\tCreate new virtual divergent loop exit "
                        << newDivergentLE->getName() << "\n");

      LLVM_DEBUG(
          dbgs() << "\tVirtual Divergent Loop Exit " << divergentLE->getName()
                 << ":\n\t\tSuccessor 0: " << target->getName()
                 << "\n\t\tSuccessor 1: " << newDivergentLE->getName() << "\n");

      auto &newDivergentLETag = DR->getOrCreateTag(newDivergentLE);

      // Set the tags.
      MaskInfos[newDivergentLE].entryMask = negCond;
      if (auto *const divLoopTag = DR->getTag(divergentLE).loop) {
        newDivergentLETag.loop = divLoopTag;
        newDivergentLETag.loop->loop->addBasicBlockToLoop(newDivergentLE, *LI);
      }

      DR->setFlag(*newDivergentLE,
                  static_cast<BlockDivergenceFlag>(
                      DR->getFlag(*divergentLE) |
                      BlockDivergenceFlag::eBlockIsVirtualDivergentLoopExit |
                      BlockDivergenceFlag::eBlockHasDivergentBranch |
                      BlockDivergenceFlag::eBlockIsDivergent));

      if (BOSCC) {
        BOSCC->addInRegions(newDivergentLE, LTag.latch);
      }

      divergentLE = newDivergentLE;
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::generateDivergentLoopResults(
    LoopTag &LTag) {
  LLVM_DEBUG(dbgs() << "CFC: GENERATE DIVERGENT LOOP RESULTS FOR "
                    << LTag.loop->getName() << "\n");

  // First create instructions to save the value of the last iteration ...
  IRBuilder<> B(LTag.header, multi_llvm::getFirstNonPHIIt(LTag.header));
  for (Value *LLV : LTag.loopLiveValues) {
    LTag.loopResultPrevs[LLV] =
        B.CreatePHI(LLV->getType(), 2, LLV->getName() + ".prev");
    LLVM_DEBUG(dbgs() << "Create result phi: "
                      << LTag.loopResultPrevs[LLV]->getName() << "\n");
  }

  // ... then create instructions to retrieve the updated value in the current
  // iteration.
  for (Value *LLV : LTag.loopLiveValues) {
    VECZ_FAIL_IF(!generateDivergentLoopResultUpdates(LLV, LTag));
  }

  if (BOSCC) {
    // Clone the loop live values update instructions in the uniform version.
    if (Loop *uniformL = BOSCC->getLoop(LTag.loop)) {
      auto *const uniformHeader = DR->getTag(uniformL).header;
      for (Value *LLV : LTag.loopLiveValues) {
        BOSCC->addReference(LTag.loopResultUpdates[LLV], LLV);
        PHINode *LRP = LTag.loopResultPrevs[LLV];
        // We only need to clone the value of the previous iteration.
        PHINode *uniformLRP = cast<PHINode>(LRP->clone());

        uniformLRP->setIncomingValue(1, LLV);

        multi_llvm::insertBefore(uniformLRP,
                                 multi_llvm::getFirstNonPHIIt(uniformHeader));
        BOSCC->createReference(LRP, uniformLRP, true);
      }
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::generateDivergentLoopResultUpdates(
    Value *LLV, LoopTag &LTag) {
  auto &LMask = LoopMasks[LTag.loop];
  Value *mask = LMask.combinedDivergentExitMask;
  VECZ_ERROR_IF(!mask, "Divergent loop does not have an exit mask");
  PHINode *PHI = LTag.loopResultPrevs[LLV];
  SelectInst *select = SelectInst::Create(
      mask, LLV, PHI, LLV->getName() + ".update", LTag.latch->getTerminator());
  LTag.loopResultUpdates[LLV] = select;

  // The PHI function of each loop live value has one incoming value from
  // the preheader if this is the outermost loop, or from the PHI function from
  // the outer loop otherwise.
  auto *const ParentL = LTag.loop->getParentLoop();
  auto *const ParentLT = ParentL ? &DR->getTag(ParentL) : nullptr;
  if (!ParentLT || !ParentLT->loopResultPrevs.count(LLV)) {
    PHI->addIncoming(getDefaultValue(PHI->getType()), LTag.preheader);
  } else {
    BasicBlock *LLVDef = cast<Instruction>(LLV)->getParent();
    if (LLVDef != LTag.header && DR->isReachable(LLVDef, LTag.header)) {
      PHI->addIncoming(LLV, LTag.preheader);
    } else {
      PHI->addIncoming(ParentLT->loopResultPrevs[LLV], LTag.preheader);
    }
  }

  LLVM_DEBUG(dbgs() << "Create result update: " << *select << "\n");

  // The second incoming value is the updated value from the latch.
  PHI->addIncoming(select, LTag.latch);

  LLVM_DEBUG(dbgs() << "Update result phi: " << *PHI << "\n");

  return true;
}

bool ControlFlowConversionState::Impl::blendDivergentLoopLiveValues(
    LoopTag &LTag, const SmallVectorImpl<BasicBlock *> &exitBlocks) {
  LLVM_DEBUG(dbgs() << "CFC: BLEND DIVERGENT LOOP LIVE VALUES FOR "
                    << LTag.loop->getName() << "\n");

  // Get the exit blocks that were not removed.
  SmallVector<BasicBlock *, 1> optionalExitBlocks;
  LTag.loop->getExitBlocks(optionalExitBlocks);
  // Remove the pure exit from it.
  for (auto it = optionalExitBlocks.begin(); it != optionalExitBlocks.end();
       ++it) {
    if (*it == LTag.pureExit) {
      (void)optionalExitBlocks.erase(it);
      break;
    }
  }

  for (Value *LLV : LTag.loopLiveValues) {
    BasicBlock *LLVDef = cast<Instruction>(LLV)->getParent();
    PHINode *prev = LTag.loopResultPrevs[LLV];
    SelectInst *update = LTag.loopResultUpdates[LLV];

    VECZ_ERROR_IF(
        !update,
        "Divergent loop live value does not have an update instruction");
    VECZ_ERROR_IF(
        !prev, "Divergent loop live value does not have a persist instruction");

    PHINode *blend =
        PHINode::Create(LLV->getType(), 2, LLV->getName() + ".blend");
    multi_llvm::insertBefore(blend, LTag.pureExit->begin());

    // Replace all uses outside the loop.
    VECZ_FAIL_IF(
        !replaceUsesOutsideDivergentLoop(LTag, LLV, blend, optionalExitBlocks));

    for (BasicBlock *EB : exitBlocks) {
      if (DR->isOptional(*EB)) {
        if (!DR->isReachable(LLVDef, EB)) {
          blend->addIncoming(prev, EB);
        } else {
          blend->addIncoming(LLV, EB);
        }
      }
    }
    blend->addIncoming(update, LTag.latch);

    if (BOSCC) {
      BOSCC->addReference(blend, update);
    }

    LLVM_DEBUG(dbgs() << "Create blend " << *blend << " for LLV " << *LLV
                      << "\n");
  }

  return true;
}

bool ControlFlowConversionState::Impl::blendDivergentLoopExitMasks(
    LoopTag &LTag, const SmallVectorImpl<Loop::Edge> &exitEdges,
    const SmallVectorImpl<BasicBlock *> &exitBlocks) {
  LLVM_DEBUG(dbgs() << "CFC: BLEND DIVERGENT LOOP EXIT MASKS FOR "
                    << LTag.loop->getName() << "\n");

  // Get the exit blocks that were not removed.
  SmallVector<BasicBlock *, 1> optionalExitBlocks;
  LTag.loop->getExitBlocks(optionalExitBlocks);
  // Remove the pure exit from it.
  for (auto it = optionalExitBlocks.begin(); it != optionalExitBlocks.end();
       ++it) {
    if (*it == LTag.pureExit) {
      (void)optionalExitBlocks.erase(it);
      break;
    }
  }

  auto &LMask = LoopMasks[LTag.loop];
  for (const Loop::Edge &EE : exitEdges) {
    BasicBlock *exitingBlock = const_cast<BasicBlock *>(EE.first);
    BasicBlock *exitBlock = const_cast<BasicBlock *>(EE.second);

    if (DR->isDivergent(*exitBlock)) {
      PHINode *prev = LMask.persistedDivergentExitMasks[exitingBlock];
      BinaryOperator *update =
          LMask.updatedPersistedDivergentExitMasks[exitingBlock];

      VECZ_ERROR_IF(
          !update,
          "Divergent loop exit mask does not have an update instruction");
      VECZ_ERROR_IF(
          !prev,
          "Divergent loop exit mask does not have a persist instruction");

      PHINode *blend =
          PHINode::Create(prev->getType(), 2, prev->getName() + ".blend");
      multi_llvm::insertBefore(blend, LTag.pureExit->begin());

      // Replace all uses outside the loop.
      VECZ_FAIL_IF(!replaceUsesOutsideDivergentLoop(LTag, update, blend,
                                                    optionalExitBlocks));

      for (BasicBlock *EB : exitBlocks) {
        if (DR->isOptional(*EB)) {
          blend->addIncoming(prev, EB);
        }
      }
      blend->addIncoming(update, LTag.latch);

      if (BOSCC) {
        BOSCC->addReference(blend, update);
      }

      LLVM_DEBUG(dbgs() << "Create blend " << *blend << " for loop exit mask "
                        << *update << "\n");
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::replaceUsesOutsideDivergentLoop(
    LoopTag &LTag, Value *from, Value *to,
    const SmallVectorImpl<BasicBlock *> &exitBlocks) {
  for (auto it = from->use_begin(); it != from->use_end();) {
    Use &U = *it++;
    Instruction *user = cast<Instruction>(U.getUser());
    BasicBlock *blockUse = user->getParent();
    // Don't replace uses within the loop.
    if (LTag.loop->contains(blockUse) ||
        // If the use is in a loop exit block, then 'to' can't reach it.
        std::count(exitBlocks.begin(), exitBlocks.end(), blockUse)) {
      continue;
    }
    // If the use is in a pure exit block of a divergent loop, don't replace
    // the use if it comes from an optional exit block of that loop.
    if (PHINode *PHI = dyn_cast<PHINode>(user)) {
      const auto *const exitedLoop = DR->getTag(blockUse).outermostExitedLoop;
      if (exitedLoop && exitedLoop->pureExit == blockUse) {
        BasicBlock *incoming = PHI->getIncomingBlock(U);
        if (!exitedLoop->loop->contains(incoming)) {
          continue;
        }
      }
    }
    U.set(to);
    LLVM_DEBUG(dbgs() << "Replace loop value " << *from << " with blend "
                      << to->getName() << "\n");
  }

  return true;
}

namespace {
using DenseDeferralMap =
    SmallDenseMap<BasicBlock *, SmallPtrSet<BasicBlock *, 2>, 32>;

void addDeferral(BasicBlock *newSrc, BasicBlock *deferred,
                 DenseDeferralMap &deferrals) {
  auto newSrcIt = deferrals.find(newSrc);
  if (newSrcIt != deferrals.end()) {
    // If the deferral edge already exists, there is no need to add it again.
    if (newSrcIt->second.count(deferred)) {
      LLVM_DEBUG(dbgs() << "\t\tDeferral (" << newSrc->getName() << ", "
                        << deferred->getName() << ") already exists\n");
      return;
    }
  }
  auto deferredIt = deferrals.find(deferred);
  if (deferredIt != deferrals.end()) {
    // If the deferral edge already exists the other way around, we don't want
    // to add it the opposite way, in risk of creating an infinite loop in the
    // CFG.
    if (deferredIt->second.count(newSrc)) {
      LLVM_DEBUG(dbgs() << "\t\tOpposite deferral (" << deferred->getName()
                        << ", " << newSrc->getName() << ") already exists\n");
      return;
    }
  }

  deferrals[newSrc].insert(deferred);

  LLVM_DEBUG(dbgs() << "\t\tAdd deferral (" << newSrc->getName() << ", "
                    << deferred->getName() << ")\n");
}

void removeDeferrals(BasicBlock *src, DenseDeferralMap &deferrals) {
  auto deferredIt = deferrals.find(src);
  if (deferredIt != deferrals.end()) {
#ifndef NDEBUG
    for (BasicBlock *deferred : deferredIt->second) {
      LLVM_DEBUG(dbgs() << "\tRemove deferral (" << src->getName() << ", "
                        << deferred->getName() << ")\n");
    }
#endif
    deferrals.erase(deferredIt);
  }
}
}  // namespace

bool ControlFlowConversionState::Impl::computeNewTargets(Linearization &lin) {
  // The entry block cannot be targeted.
  const auto &DCBI = DR->getBlockOrdering();
  const size_t numBlocks = DCBI.size();
  DenseSet<BasicBlock *> targets(numBlocks - 1);
  for (const auto &tag : make_range(std::next(DCBI.begin()), DCBI.end())) {
    targets.insert(tag.BB);
  }

  DenseDeferralMap deferrals;

  LLVM_DEBUG(dbgs() << "CFC: COMPUTE NEW TARGETS\n");

  // For each basic block, select its new targets based on previous blocks that
  // have been deferred because of divergence, and their current successors.
  // Select the target that has the lowest DCBI, i.e. the block whose dominance
  // englobes or is equal to the other available targets.
  //
  // If we assign a target different from the current successor of the block,
  // we must add a deferral edge from the selected target to the current
  // successor (that got replaced by the selected target) such that an edge
  // from the current block to the replaced successor exists in the modified
  // graph.
  lin.infos.reserve(numBlocks);
  lin.data.reserve(numBlocks);
  for (size_t BBIndex = 0; BBIndex != numBlocks; ++BBIndex) {
    const auto &BBTag = DR->getBlockTag(BBIndex);
    BasicBlock *BB = BBTag.BB;
    lin.beginBlock(BB);

    LLVM_DEBUG(dbgs() << "BB " << BB->getName() << ":\n");

    // Retrieve the rewire list for 'BB'.
    SmallPtrSet<BasicBlock *, 8> availableTargets;
    {
      auto deferredIt = deferrals.find(BB);
      if (deferredIt != deferrals.end()) {
        for (BasicBlock *deferred : deferredIt->second) {
          availableTargets.insert(deferred);
        }
      }
    }

    if (!DR->isDivCausing(*BB) ||
        // Loop latches must have their branch retained.
        (BBTag.loop && BBTag.loop->latch == BB)) {
      // If 'BB' ends in a uniform branch.
      LLVM_DEBUG(dbgs() << "  uniform branch\n");

      // Keep track of what blocks we have targeted in case we have a deferred
      // block that is a current successor (which could lead in choosing the
      // same block twice!).
      SmallPtrSet<BasicBlock *, 8> targeted;

      for (BasicBlock *succ : successors(BB)) {
        size_t nextIndex = ~size_t(0);
        for (BasicBlock *deferred : availableTargets) {
          if (targeted.count(deferred)) {
            continue;
          }

          const size_t deferredIndex = DR->getTagIndex(deferred);
          if (nextIndex == ~size_t(0) || nextIndex > deferredIndex) {
            nextIndex = deferredIndex;
          }
        }

        const size_t succIndex = DR->getTagIndex(succ);
        if (!targeted.count(succ)) {
          // If we have not found a target or there is a better one.
          if (nextIndex == ~size_t(0) || nextIndex > succIndex) {
            nextIndex = succIndex;
          }
        }

        VECZ_ERROR_IF(nextIndex == ~size_t(0), "No target was found");

        auto *const next = DR->getBlockTag(nextIndex).BB;
        lin.push(next);
        targeted.insert(next);

        LLVM_DEBUG(dbgs() << "\tsuccessor " << lin.currentSize() - 1 << ": "
                          << next->getName() << "\n");

        // Virtually remove backedges.
        if (!BBTag.isLoopBackEdge(next)) {
          targets.erase(next);
          // Don't add deferred edges to blocks already processed.
          if (BBIndex < nextIndex) {
            auto S = availableTargets;
            S.insert(succ);

            for (BasicBlock *deferred : S) {
              if (deferred != next) {
                addDeferral(next, deferred, deferrals);
              }
            }
          }
        }
      }
    } else {
      LLVM_DEBUG(dbgs() << "  divergent branch\n");

      for (BasicBlock *succ : successors(BB)) {
        availableTargets.insert(succ);
      }

      size_t nextIndex = ~size_t(0);
      for (BasicBlock *deferred : availableTargets) {
        const size_t deferredIndex = DR->getTagIndex(deferred);
        if (nextIndex == ~size_t(0) || nextIndex > deferredIndex) {
          LLVM_DEBUG(dbgs()
                     << (nextIndex == ~size_t(0)
                             ? "\tchoosing successor: "
                             : "\tpreferring instead successor: ")
                     << DR->getBlockTag(deferredIndex).BB->getName() << "\n");
          nextIndex = deferredIndex;
        }
      }

      VECZ_ERROR_IF(nextIndex == ~size_t(0), "No target was found");

      BasicBlock *const next = DR->getBlockTag(nextIndex).BB;
      lin.push(next);

      // The last eBlockIsVirtualDivergentLoopExit introduced from an optional
      // loop exit wasn't given a block to branch to, it is thus empty.
      if (DR->hasFlag(*BB,
                      BlockDivergenceFlag::eBlockIsVirtualDivergentLoopExit) &&
          !BB->getTerminator()) {
        BranchInst::Create(next, BB);
      }

      LLVM_DEBUG(dbgs() << "\tsuccessor 0: " << next->getName() << "\n");

      // Virtually remove backedges.
      if (!BBTag.isLoopBackEdge(next)) {
        targets.erase(next);
        // Don't add deferred edges to blocks already processed.
        if (BBIndex < nextIndex) {
          for (BasicBlock *deferred : availableTargets) {
            if (deferred != next) {
              addDeferral(next, deferred, deferrals);
            }
          }
        }
      }
    }

    // Remove the deferrals that involved 'BB'.
    removeDeferrals(BB, deferrals);

    // clang-format off
    LLVM_DEBUG(
        dbgs() << "  deferral list:";
        if (deferrals.empty()) {
          dbgs() << " (empty)\n";
        } else {
          dbgs() << "\n";
          for (const auto &pair : deferrals) {
            for (BasicBlock *deferred : pair.second) {
              LLVM_DEBUG(dbgs() << "\t(" << pair.first->getName() << ", "
                                << deferred->getName() << ")\n");
            }
          }
        }
    );
    // clang-format on
  }

  // There shouldn't remain any deferral edges.
  VECZ_ERROR_IF(!deferrals.empty(), "Deferrals remain");
  // All blocks should have been targeted at least once.
  VECZ_ERROR_IF(!targets.empty(), "Not all blocks have been rewired");

  return true;
}

bool ControlFlowConversionState::Impl::linearizeCFG() {
  LLVM_DEBUG(dbgs() << "CFC: LINEARIZE\n");

  // Compute the new targets ...
  Linearization lin;
  VECZ_FAIL_IF(!computeNewTargets(lin));

  auto dataIt = lin.data.begin();
  for (const auto &newTargetInfo : lin.infos) {
    BasicBlock *BB = newTargetInfo.BB;

    // Get the new target info for this block
    const auto numTargets = newTargetInfo.numTargets;
    const auto newTargets = dataIt;
    dataIt += numTargets;

    LLVM_DEBUG(dbgs() << BB->getName() << ":\n");

    auto *T = BB->getTerminator();

    // If we have set a new target that is already a successor of BB, but we
    // have not set it at the same successor's position, then do it!
    // It will avoid to have to update the phi nodes.
    SmallDenseMap<BasicBlock *, unsigned, 2> successors;
    for (unsigned idx = 0; idx < T->getNumSuccessors(); ++idx) {
      BasicBlock *succ = T->getSuccessor(idx);
      successors.try_emplace(succ, idx);
    }
    for (unsigned idx = 0; idx < numTargets; ++idx) {
      auto succIt = successors.find(newTargets[idx]);
      // If we have a successor set as a new target ...
      if (succIt != successors.end()) {
        // ... but we have not set it at the same position ...
        if (succIt->second != idx && succIt->second < numTargets) {
          // .. then swap both blocks.
          std::swap(newTargets[idx], newTargets[succIt->second]);
        }
      }
    }

    // Now iterate over the new targets to set them as successors of BB if
    // they were not already.
    unsigned idx = 0;
    for (; idx < numTargets; ++idx) {
      BasicBlock *const newTarget = newTargets[idx];

      VECZ_ERROR_IF(
          idx >= T->getNumSuccessors(),
          "BasicBlock should not have more successors after linearization");

      BasicBlock *oldSucc = T->getSuccessor(idx);

      LLVM_DEBUG(dbgs() << "\tOld successor: " << oldSucc->getName() << "\n");

      // If we have set the current successor to be the new target, there is
      // nothing to do.
      if (oldSucc == newTarget) {
        LLVM_DEBUG(dbgs() << "\tUntouched successor: " << oldSucc->getName()
                          << "\n");
        continue;
      }

      // Uniform blocks should not be rewired.
      VECZ_ERROR_IF(DR->isUniform(*oldSucc),
                    "Uniform BasicBlock should not have its edge modified");

      // Otherwise update the successor.
      T->setSuccessor(idx, newTarget);
      LLVM_DEBUG(dbgs() << "\tAdd successor: " << newTarget->getName() << "\n");
    }

    // We have either processed a divergent branch (with only one successor), or
    // we have processed a uniform branch (with all its successors untouched).
    VECZ_ERROR_IF(idx != 1 && idx != T->getNumSuccessors(),
                  "Number of processed new targets is undefined");

    // Finally, clear the remaining successors that have not been set as new
    // targets.
    if (idx != T->getNumSuccessors()) {
      for (; idx < T->getNumSuccessors(); ++idx) {
        BasicBlock *succ = T->getSuccessor(idx);

        // Uniform blocks should not be rewired.
        VECZ_ERROR_IF(DR->isUniform(*succ),
                      "Uniform BasicBlock should not have its edge modified");

        LLVM_DEBUG(dbgs() << "\tRemove successor: " << succ->getName() << "\n");
      }

      auto *newT = BranchInst::Create(T->getSuccessor(0), T);

      updateMaps(T, newT);

      IRCleanup::deleteInstructionNow(T);
    }
  }
  assert(dataIt == lin.data.end() &&
         "Failed to reach end of Linearization data vector!");

  // Updating on-the-fly the DomTree and PostDomTree whilst rewiring the CFG
  // is extremely tedious, and may not even be possible due to all the invalid
  // states that happen during it ... Therefore, we have no choice but to
  // recalculate the DomTree and PostDomTree from scratch.
  DT->recalculate(F);
  PDT->recalculate(F);
  VECZ_ERROR_IF(!DT->verify(), "DominatorTree incorrectly updated");
  VECZ_ERROR_IF(!PDT->verify(), "PostDominatorTree incorrectly updated");
  VECZ_FAIL_IF(!computeBlockOrdering());
  RC->clear();

  return true;
}

bool ControlFlowConversionState::Impl::generateSelects() {
  LLVM_DEBUG(dbgs() << "CFC: GENERATE SELECTS FROM PHI NODES\n");
  // For each basic block that has only one predecessor and phi nodes, we need
  // to either blend those phi nodes into select instructions or try to move
  // the phi nodes up the chain of linearized path.
  for (const auto &BTag : DR->getBlockOrdering()) {
    BasicBlock *B = BTag.BB;
    if (B->hasNPredecessors(1) || DR->isBlend(*B)) {
      if (PHINode *PHI = dyn_cast<PHINode>(&B->front())) {
        LLVM_DEBUG(dbgs() << B->getName() << ":\n");
        const SmallPtrSet<BasicBlock *, 2> incomings(PHI->block_begin(),
                                                     PHI->block_end());
        BasicBlock *cur = B;
        while (cur->hasNPredecessors(1) && !incomings.empty()) {
          cur = cur->getSinglePredecessor();
          if (incomings.count(cur)) {
            break;
          }
        }
        // Only move the phis up the chain of linearized path:
        // - if the block whose phis we are processing is not a blend block
        //   (because the latter do need to have its phis transformed into
        //   selects),
        // - if the last block of the chain is not an incoming block, and
        // - if the last block of the chain is a convergence block.
        if (!DR->isBlend(*B) && !incomings.count(cur) &&
            cur->hasNPredecessorsOrMore(2) && PHI->getNumIncomingValues() > 1) {
          // All PHI nodes have the same incoming blocks so we update the exit
          // masks of the incoming blocks of the first PHI node here.
          for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
            auto &maskInfo = MaskInfos[PHI->getIncomingBlock(i)];
            Value *&exitMask = maskInfo.exitMasks[cur];

            if (!exitMask) {
              exitMask = maskInfo.exitMasks[B];
            }
          }

          while ((PHI = dyn_cast<PHINode>(&B->front()))) {
            LLVM_DEBUG(dbgs() << "\tMove " << *PHI << " in " << cur->getName()
                              << "\n");
            PHI->moveBefore(*cur, cur->begin());
          }
        } else {
          while ((PHI = dyn_cast<PHINode>(&B->front()))) {
            VECZ_FAIL_IF(!generateSelectFromPHI(PHI, B));
          }
        }
      }
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::generateSelectFromPHI(PHINode *PHI,
                                                             BasicBlock *B) {
  const unsigned phiNumIncVals = PHI->getNumIncomingValues();
  VECZ_ERROR_IF(phiNumIncVals == 0, "PHINode does not have any incoming value");

  Value *newVal = nullptr;
  auto &maskInfo = MaskInfos[B];
  if (PHI == maskInfo.entryMask) {
    // The entry mask of a blend value should be the conjunction of the incoming
    // masks, so change it.
    maskInfo.entryMask = copyEntryMask(PHI->getIncomingValue(0), *B);
    for (unsigned i = 1; i < phiNumIncVals; i++) {
      Value *V = PHI->getIncomingValue(i);
      Instruction *insertBefore =
          cast<Instruction>(maskInfo.entryMask)->getNextNode();
      maskInfo.entryMask = BinaryOperator::CreateOr(
          maskInfo.entryMask, V, B->getName() + ".entry_mask", insertBefore);
    }
    newVal = maskInfo.entryMask;
  } else {
    Value *select = PHI->getIncomingValue(0);
    for (unsigned i = 1; i < phiNumIncVals; i++) {
      Value *V = PHI->getIncomingValue(i);
      BasicBlock *PHIB = PHI->getIncomingBlock(i);
      Value *cond = MaskInfos[PHIB].exitMasks[B];
      VECZ_ERROR_IF(!cond, "Exit mask does not exist");

      Instruction *insertBefore = &*B->getFirstInsertionPt();
      if (i == 1) {
        if (Instruction *condI = dyn_cast<Instruction>(cond)) {
          BasicBlock *maskParent = condI->getParent();
          if (maskParent == B) {
            insertBefore = condI->getNextNode();
          }
        }
      } else {
        insertBefore = cast<Instruction>(select)->getNextNode();
      }
      select = SelectInst::Create(cond, V, select, PHI->getName() + ".blend",
                                  insertBefore);
    }
    newVal = select;
  }

  LLVM_DEBUG(dbgs() << "\tReplace " << *PHI << " with " << *newVal << "\n");

  updateMaps(PHI, newVal);

  PHI->replaceAllUsesWith(newVal);

  IRCleanup::deleteInstructionNow(PHI);

  return true;
}

bool ControlFlowConversionState::Impl::repairSSA() {
  // Check that all the blocks have a unique position
  VECZ_FAIL_IF(!checkBlocksOrder());
  RC->update(F);

  VECZ_FAIL_IF(!updatePHIsIncomings());
  VECZ_FAIL_IF(!blendInstructions());

  VECZ_ERROR_IF(!DT->verify(), "DominatorTree incorrectly updated");
  VECZ_ERROR_IF(!PDT->verify(), "PostDominatorTree incorrectly updated");

  return true;
}

bool ControlFlowConversionState::Impl::updatePHIsIncomings() {
  // We need to update the incoming blocks of phi nodes whose predecessors may
  // have changed since we have not changed the phi nodes during the rewiring.
  for (const auto &BBTag : DR->getBlockOrdering()) {
    BasicBlock *BB = BBTag.BB;
    const SmallPtrSet<BasicBlock *, 4> preds(pred_begin(BB), pred_end(BB));
    for (auto it = BB->begin(); it != BB->end();) {
      Instruction &I = *it++;
      PHINode *PHI = dyn_cast<PHINode>(&I);
      if (!PHI) {
        break;
      }

      const SmallPtrSet<BasicBlock *, 4> incomings(PHI->block_begin(),
                                                   PHI->block_end());

      // If no predecessors of `BB` is an incoming block of its PHI Node, then
      // completely transform the PHI Node into multiple select instructions.
      bool intersect = false;
      for (BasicBlock *inc : incomings) {
        for (BasicBlock *pred : preds) {
          if (pred == inc) {
            intersect = true;
            break;
          }
        }
        if (intersect) {
          break;
        }
      }
      if (!intersect) {
        VECZ_FAIL_IF(!generateSelectFromPHI(PHI, BB));
        continue;
      }
      // Otherwise, only transform the incoming blocks of predecessors that got
      // linearized into selects.
      //
      // Instruction that will combine the phi node and the select instructions
      // created from it if some incoming blocks are no longer predecessors.
      Instruction *newBlend = nullptr;
      Instruction *insertBefore = getInsertionPt(*BB);

      auto &maskInfo = MaskInfos[BB];
      const bool isEntryMask = PHI == maskInfo.entryMask;
      for (unsigned idx = 0; idx < PHI->getNumIncomingValues(); ++idx) {
        BasicBlock *incoming = PHI->getIncomingBlock(idx);
        if (preds.count(incoming)) {
          continue;
        }
        // If the incoming block is no longer a predecessor, transform it into
        // a select instruction, or a binary OR if it is an entry mask.
        Value *V = PHI->getIncomingValue(idx);

        if (isEntryMask) {
          // The entry mask of a blend value should be the conjunction of
          // the incoming masks, so change it.
          if (!newBlend) {
            newBlend = BinaryOperator::CreateOr(
                PHI, V, BB->getName() + ".entry_mask", insertBefore);
          } else {
            newBlend = BinaryOperator::CreateOr(
                newBlend, V, BB->getName() + ".entry_mask", insertBefore);
          }
          maskInfo.entryMask = newBlend;
        } else {
          Value *cond = MaskInfos[incoming].exitMasks[BB];
          VECZ_ERROR_IF(!cond, "Exit mask does not exist");
          if (!newBlend) {
            newBlend = SelectInst::Create(
                cond, V, PHI, PHI->getName() + ".blend", insertBefore);
          } else {
            newBlend = SelectInst::Create(
                cond, V, newBlend, PHI->getName() + ".blend", insertBefore);
          }
        }
        PHI->removeIncomingValue(idx--);
      }

      // If we have created select instructions from `PHI`, update the users
      // of the latter.
      if (newBlend) {
        VECZ_FAIL_IF(!replaceReachableUses(*RC, PHI, newBlend, BB));
        updateMaps(PHI, newBlend);
      }

      // And add any new incoming blocks that do not replace any previous.
      for (BasicBlock *pred : preds) {
        if (!incomings.count(pred)) {
          PHI->addIncoming(getDefaultValue(PHI->getType()), pred);
        }
      }
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::blendInstructions() {
  LLVM_DEBUG(dbgs() << "CFC: BLEND INSTRUCTIONS\n");

  auto addSuccessors = [this](const BasicBlockTag &BTag, BlockQueue &queue,
                              DenseSet<BasicBlock *> &visited,
                              const BasicBlockTag &dstTag) {
    for (BasicBlock *succ : successors(BTag.BB)) {
      // Allow latch if 'succ' belongs in 'dst's loop and 'dst' is the header
      // of that loop.
      const bool allowLatch =
          dstTag.isLoopHeader() && dstTag.loop->loop->contains(succ);

      if (!allowLatch && BTag.isLoopBackEdge(succ)) {
        continue;
      }

      if (allowLatch) {
        // the fast Reachability calculation can't follow back edges yet
        if (!DR->isReachable(succ, dstTag.BB, allowLatch)) {
          continue;
        }
      } else if (!RC->isReachable(succ, dstTag.BB)) {
        continue;
      }

      if (visited.insert(succ).second) {
        LLVM_DEBUG(dbgs() << "\t\t\tInsert " << succ->getName()
                          << " in the queue\n");
        queue.push(DR->getTagIndex(succ));
      }
    }

    // clang-format off
    LLVM_DEBUG(
        dbgs() << "\t\t\tWorklist: [";
        if (!queue.empty()) {
          dbgs() << DR->getBlockTag(*queue.begin()).BB->getName();
          for (auto it = std::next(queue.begin()); it != queue.end(); ++it) {
            dbgs() << ", " << DR->getBlockTag(*it).BB->getName();
          }
          dbgs() << "]\n";
        }
    );
    // clang-format on
  };

  DenseMap<Instruction *, SmallDenseMap<BasicBlock *, Value *, 2>> blendMap;

  auto getValueOfAt = [&blendMap](Instruction *opDef,
                                  BasicBlock *B) -> Value * {
    auto it = blendMap.find(opDef);
    if (it != blendMap.end()) {
      auto it2 = it->second.find(B);
      if (it2 != it->second.end()) {
        return it2->second;
      }
    }
    return nullptr;
  };

  auto createBlend = [this, &blendMap, &getValueOfAt](
                         BasicBlock *B, Instruction *opDef) -> Value * {
    if (Value *V = getValueOfAt(opDef, B)) {
      return V;
    }

    Type *T = opDef->getType();
    const unsigned numPreds = std::distance(pred_begin(B), pred_end(B));
    Value *blend = nullptr;
    PHINode *PHI = PHINode::Create(T, numPreds, opDef->getName() + ".merge");
    multi_llvm::insertBefore(PHI, B->begin());

    const auto *const LTag = DR->getTag(B).loop;
    bool hasVisitedPred = false;
    for (BasicBlock *pred : predecessors(B)) {
      Value *incomingV = nullptr;
      if (Value *predV = getValueOfAt(opDef, pred)) {
        incomingV = predV;
        hasVisitedPred = true;
      } else {
        // When blending a loop header, the value coming from the latch should
        // be the one coming from the preheader if that value dominates the
        // latch and the latch has no definition of the value we are trying to
        // blend.
        if (DR->getTag(pred).isLoopBackEdge(B)) {
          if (Value *preheaderV = getValueOfAt(opDef, LTag->preheader)) {
            if (auto *instV = dyn_cast<Instruction>(preheaderV)) {
              if (DT->dominates(instV->getParent(), pred)) {
                incomingV = preheaderV;
              }
            } else {
              incomingV = preheaderV;
            }
          }
        }
      }

      if (!incomingV) {
        incomingV = getDefaultValue(T);
      }
      PHI->addIncoming(incomingV, pred);
    }
    if (!hasVisitedPred) {
      IRCleanup::deleteInstructionNow(PHI);
      return nullptr;
    }

    if (PHI->hasConstantValue()) {
      blend = PHI->getIncomingValue(0);
      IRCleanup::deleteInstructionNow(PHI);
    } else {
      blend = PHI;
      blends.insert(PHI);
    }

    blendMap[opDef][B] = blend;

    return blend;
  };

  // Manually set the entry point of persisted loop live values and persisted
  // loop exit masks.
  for (auto *const LTag : DR->getLoopOrdering()) {
    auto *const header = LTag->header;
    for (Value *LLV : LTag->loopLiveValues) {
      Instruction *LLVI = cast<Instruction>(LLV);
      if (LLVI->getParent() != header) {
        blendMap[LLVI][header] = LTag->loopResultPrevs[LLV];
      }
    }

    auto &LMask = LoopMasks[LTag->loop];
    for (auto &UPREM : LMask.updatedPersistedDivergentExitMasks) {
      if (UPREM.first != header) {
        blendMap[UPREM.second][header] =
            LMask.persistedDivergentExitMasks[UPREM.first];
      }
    }
  }

  SmallPtrSet<Value *, 16> spareBlends;

  for (const auto &dstTag : DR->getBlockOrdering()) {
    BasicBlock *dst = dstTag.BB;
    LLVM_DEBUG(dbgs() << "Blending instructions used in " << dst->getName()
                      << ":\n");
    for (Instruction &I : *dst) {
      // Don't try to blend a blend value.
      if (blends.count(&I)) {
        continue;
      }

      LLVM_DEBUG(dbgs() << "\tInstruction " << I << ":\n");

      for (unsigned idx = 0; idx < I.getNumOperands(); ++idx) {
        Instruction *opDef = dyn_cast<Instruction>(I.getOperand(idx));
        if (!opDef) {
          continue;
        }

        BasicBlock *src = opDef->getParent();

        LLVM_DEBUG(dbgs() << "\t\tOperand " << *opDef << "\n\t\tdefined in "
                          << src->getName() << ":\n");

        blendMap[opDef][src] = opDef;

        // There exists two possible ways to early exit the blend instruction:
        // - if the current block dominates the 'dst'.
        // - if the current block dominates the incoming block of the phi node
        //   'I' we are blending in 'dst'.
        //
        // 'dst' can freely access the values of 'src'.
        if (DT->dominates(src, dst)) {
          LLVM_DEBUG(dbgs() << "\t\t\tDefinition dominates use\n");
          continue;
        }
        // The incoming block of this phi node is dominated by the definition
        // block of the incoming value.
        BasicBlock *incoming = nullptr;
        if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
          incoming = PHI->getIncomingBlock(idx);
          if (DT->dominates(src, incoming)) {
            LLVM_DEBUG(dbgs() << "\t\t\tDefinition dominates use\n");
            continue;
          }
        }

        DenseSet<BasicBlock *> visited;
        BlockQueue queue(*DR);

        const auto &srcTag = DR->getTag(src);

        addSuccessors(srcTag, queue, visited, dstTag);

        auto *const srcLoop = srcTag.loop;
        if (srcLoop && srcLoop->isLoopDivergent()) {
          if (dst != srcLoop->header) {
            auto &srcMasks = LoopMasks[srcLoop->loop];
            const auto &headerTag = DR->getTag(srcLoop->header);

            // If 'opDef' is an update loop exit mask, set an entry point in
            // the loop header.
            auto UPREMIt =
                srcMasks.updatedPersistedDivergentExitMasks.find(src);
            if (UPREMIt != srcMasks.updatedPersistedDivergentExitMasks.end()) {
              if (UPREMIt->second == opDef) {
                LLVM_DEBUG(dbgs()
                           << "\t\t\tFound persisted value of the operand: "
                           << srcMasks.persistedDivergentExitMasks[src]
                           << "\n");
                addSuccessors(headerTag, queue, visited, dstTag);
              }
            }
            // If 'opDef' is a loop live value, set an entry point in the loop
            // header.
            if (srcLoop->loopLiveValues.count(opDef)) {
              LLVM_DEBUG(dbgs()
                         << "\t\t\tFound persisted value of the operand: "
                         << srcLoop->loopResultPrevs[opDef] << "\n");
              addSuccessors(headerTag, queue, visited, dstTag);
            }
          }
        }

        while (!queue.empty()) {
          const BasicBlockTag &curTag = queue.pop();
          BasicBlock *const cur = curTag.BB;

          LLVM_DEBUG(dbgs() << "\t\t\tPopping " << cur->getName() << "\n");

          // We have reached 'dst' without finding a block that dominates it,
          // we need to create a phi node if the user is not one, and replace
          // the operand with the last blended value.
          if (cur == dst) {
            LLVM_DEBUG(dbgs() << "\t\t\tReached destination: ");
            VECZ_ERROR_IF(!queue.empty(), "Blocks remain in the queue");
            if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
              BasicBlock *incoming = PHI->getIncomingBlock(idx);
              Value *V = getValueOfAt(opDef, incoming);
              VECZ_ERROR_IF(!V, "No blend value was found");
              I.setOperand(idx, V);
            } else {
              Value *blend = createBlend(cur, opDef);
              VECZ_ERROR_IF(!blend, "No blend value was found");
              spareBlends.erase(blend);
              I.setOperand(idx, blend);
            }
            LLVM_DEBUG(dbgs() << "new operand: " << *I.getOperand(idx) << "\n");
            break;
          }

          const bool curDomDst = DT->dominates(cur, dst);
          const bool curDomInc = incoming && DT->dominates(cur, incoming);
          const bool srcDomCur = DT->dominates(src, cur);

          auto &opDefBlend = blendMap[opDef];
          // If either condition is true, we can early exit:
          // - 'dst' can freely access the values of 'cur',
          // - 'incoming' can freely access the values of 'cur'.
          if ((curDomDst || curDomInc) && queue.empty()) {
            LLVM_DEBUG(dbgs() << "\t\t\tBlock " << cur->getName()
                              << " dominates destination: ");
            if (srcDomCur) {
              auto *const blend = opDefBlend[src];
              opDefBlend[cur] = blend;
              I.setOperand(idx, blend);
            } else {
              auto *const blend = createBlend(cur, opDef);
              VECZ_ERROR_IF(!blend, "No blend value was found");
              spareBlends.erase(blend);
              I.setOperand(idx, blend);
            }
            LLVM_DEBUG(dbgs() << "new operand: " << *I.getOperand(idx) << "\n");
            break;
          }

          addSuccessors(curTag, queue, visited, dstTag);

          // 'cur' can freely access 'opDef'.
          if (srcDomCur) {
            // DANGER! operator[] returns a reference, which may be invalidated
            // by a second call to it. Therefore we have to copy the value via
            // a temporary variable.
            auto *const blendSrc = opDefBlend[src];
            opDefBlend[cur] = blendSrc;
            continue;
          }

          // 'cur' does not have a blend value of 'opDef' so create one.
          Value *blend = createBlend(cur, opDef);
          VECZ_ERROR_IF(!blend, "No blend value was found");
          if (isa<PHINode>(blend)) {
            spareBlends.insert(blend);
          }
        }
      }
    }
  }

  for (auto *blend : spareBlends) {
    auto *I = cast<Instruction>(blend);
    if (I->use_empty()) {
      IRCleanup::deleteInstructionNow(I);
    }
  }

  return true;
}

bool ControlFlowConversionState::Impl::simplifyMasks() {
  const SimplifyQuery Q(F.getParent()->getDataLayout(), nullptr, DT);

  // We might like to just look at the masks pointed to by the block/loop tags,
  // however linearization and/or BOSCC can sometimes delete them from under
  // our nose so it's only safe just to go through all the boolean operations
  // and see if we can simplify any of them.
  for (const auto &BBTag : DR->getBlockOrdering()) {
    SmallVector<Instruction *, 16> toDelete;
    for (auto &I : *BBTag.BB) {
      if (isa<SelectInst>(&I) || (I.getType()->getScalarSizeInBits() == 1 &&
                                  (isa<BinaryOperator>(&I) ||
                                   isa<PHINode>(&I) || isa<ICmpInst>(&I)))) {
        if (I.use_empty()) {
          toDelete.push_back(&I);
        } else {
          Value *simpleMask = simplifyInstruction(&I, Q);
          if (simpleMask && simpleMask != &I) {
            I.replaceAllUsesWith(simpleMask);
            toDelete.push_back(&I);
          }
        }
      }
    }
    for (auto *I : toDelete) {
      IRCleanup::deleteInstructionNow(I);
    }
  }

  return true;
}

bool ControlFlowConversionState::computeBlockOrdering() {
  LLVM_DEBUG(dbgs() << "CFC: COMPUTE BLOCK ORDERING\n");
  RC->clear();
  return DR->computeBlockOrdering(*DT);
}

bool ControlFlowConversionState::Impl::checkBlocksOrder() const {
  const auto &DCBI = DR->getBlockOrdering();
  VECZ_ERROR_IF(F.size() != DCBI.size(),
                "Worklist does not contain all blocks");

  uint32_t next = 0u;
  for (const auto &BBTag : DCBI) {
    VECZ_ERROR_IF(BBTag.pos != next,
                  "BasicBlock indices not in consecutive order");
    ++next;
  }

  return true;
}

void ControlFlowConversionState::Impl::updateMaps(Value *from, Value *to) {
  // Because we keep track of mapping values between uniform and predicated
  // version, since we replace 'from' with 'to', we also have to update
  // the hashtable.
  if (BOSCC) {
    BOSCC->updateValue(from, to);
  }

  // Because we keep track of loop live values, since we replace 'from' with
  // 'to', we also have to update the hashset.
  for (auto *const LTag : DR->getLoopOrdering()) {
    if (LTag->loopLiveValues.erase(from)) {
      LTag->loopLiveValues.insert(to);
      auto LRPIt = LTag->loopResultPrevs.find(from);
      if (LRPIt != LTag->loopResultPrevs.end()) {
        PHINode *from = LRPIt->second;
        LTag->loopResultPrevs.erase(LRPIt);
        LTag->loopResultPrevs[to] = from;
      }
      auto LRUIt = LTag->loopResultUpdates.find(from);
      if (LRUIt != LTag->loopResultUpdates.end()) {
        SelectInst *select = LRUIt->second;
        LTag->loopResultUpdates.erase(LRUIt);
        LTag->loopResultUpdates[to] = select;
      }
    }
  }
}
