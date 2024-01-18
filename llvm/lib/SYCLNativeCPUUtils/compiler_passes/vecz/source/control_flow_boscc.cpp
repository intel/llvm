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

#include "control_flow_boscc.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <multi_llvm/multi_llvm.h>

#include <numeric>
#include <queue>
#include <utility>

#include "analysis/divergence_analysis.h"
#include "analysis/liveness_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "llvm_helpers.h"
#include "reachability.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"

#define DEBUG_TYPE "vecz-cf"

using namespace llvm;
using namespace vecz;

namespace {
using RPOT = ReversePostOrderTraversal<Function *>;

bool isUsedOutsideDefinitionBlock(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    return std::any_of(I->user_begin(), I->user_end(), [&I](User *U) {
      return cast<Instruction>(U)->getParent() != I->getParent();
    });
  }
  return false;
}

/// @brief Check whether a block is "trivial" according to a heuristic
/// @param[in] BB the Basic Block to check
/// @return true if the block is trivial
bool isTrivialBlock(const BasicBlock &BB) {
  if (BB.size() > 3) {
    return false;
  }

  for (const auto &I : BB) {
    if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() ||
        isa<PHINode>(&I)) {
      return false;
    }
  }
  return true;
}

}  // namespace

/// @brief Check whether a uniform region is viable and worth keeping.
/// @param[in] region the region to check
/// @param[in] noDuplicateBlocks blocks the region is not alowed to contain
/// @return false iff the region should be discarded.

bool ControlFlowConversionState::BOSCCGadget::duplicateUniformRegions() {
  LLVM_DEBUG(dbgs() << "DUPLICATE UNIFORM REGIONS\n");

  // Keep tracks of blocks that contain NoDuplicate calls.
  DenseSet<BasicBlock *> noDuplicateBlocks;
  SmallPtrSet<Loop *, 16> noDuplicateLoops;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (CI->hasFnAttr(Attribute::NoDuplicate)) {
          noDuplicateBlocks.insert(&BB);
          auto *const loop = DR->getTag(&BB).loop;
          if (loop) {
            noDuplicateLoops.insert(loop->loop);
          }
          break;
        }
      }
    }
  }

  // First, create the regions.
  VECZ_FAIL_IF(!createUniformRegions(noDuplicateBlocks));

  // Keep track of blocks that belong to loops. If a whole loop is duplicated,
  // then a new loop object should be created for the uniform version.
  SmallVector<Loop *, 16> duplicatedLoops;
  SmallPtrSet<Loop *, 16> duplicatedLoopSet;

  const size_t size =
      std::accumulate(uniformRegions.begin(), uniformRegions.end(), 0,
                      [](size_t base, const UniformRegion &region) {
                        return base + region.predicatedBlocks.size();
                      });
  std::vector<BasicBlock *> newBlocks;
  newBlocks.reserve(size);

  // Conserve the original edges of the CFG.
  for (BasicBlock &BB : F) {
    for (BasicBlock *succ : successors(&BB)) {
      uniformEdges[&BB].push_back(succ);
    }
  }

  // Then duplicate them.
  for (auto &region : uniformRegions) {
    BasicBlock *entry = region.entryBlock;

    std::vector<BasicBlock *> sortedNewRegionBlocks;
    sortedNewRegionBlocks.reserve(region.predicatedBlocks.size());

    // Process the region's predicated blocks in DCBI order.
    // Gather the block indices, then sort them.
    std::vector<size_t> predicatedBlockIndices;
    predicatedBlockIndices.reserve(region.predicatedBlocks.size());
    for (auto *const B : region.predicatedBlocks) {
      predicatedBlockIndices.push_back(DR->getTagIndex(B));
    }
    std::sort(predicatedBlockIndices.begin(), predicatedBlockIndices.end());

    for (const auto index : predicatedBlockIndices) {
      const auto &BTag = DR->getBlockTag(index);
      auto *const B = BTag.BB;
      auto *const LTag = BTag.loop;

      // If the block is the BOSCC entry block, we don't want to duplicate it
      // unless it is part of a loop.
      if (B == entry && !LTag) {
        continue;
      }

      BasicBlock *newB = nullptr;
      // If we have already cloned 'B', then we can reuse the cloned version.
      if (VMap.count(B)) {
        continue;
      }

      newB = CloneBasicBlock(B, VMap, ".uniform", &F);
      VMap.insert({B, newB});
      region.uniformBlocks.insert(newB);
      newBlocks.push_back(newB);
      sortedNewRegionBlocks.push_back(newB);

      // The new blocks will remain uniform
      BasicBlockTag &newBTag = DR->getOrCreateTag(newB);
      DR->setFlag(*newB, eBlockIsUniform);

      if (LTag) {
        auto *const loop = LTag->loop;
        if (LTag->header == B) {
          duplicatedLoopSet.insert(loop);
          duplicatedLoops.push_back(loop);
        }

        if (!duplicatedLoopSet.count(loop)) {
          newBTag.loop = LTag;
          loop->addBasicBlockToLoop(newB, *LI);
        }
      }
    }

    // Splice the newly inserted blocks into the function right before the
    // first div_causing block.
    if (!sortedNewRegionBlocks.empty() &&
        entry->getNextNode() != sortedNewRegionBlocks[0]) {
      F.splice(entry->getNextNode()->getIterator(), &F,
               sortedNewRegionBlocks[0]->getIterator(), F.end());
    }
  }

  // Since we added all loops by their headers in DCBI order, inner loops will
  // always follow outer loops, so there is no need to sort them.
  for (Loop *L : duplicatedLoops) {
    if (!LMap.count(L) && !noDuplicateLoops.count(L)) {
      VECZ_FAIL_IF(!duplicateUniformLoops(L));
    }
  }

  // Fix the duplicated instructions arguments.
  for (BasicBlock *B : newBlocks) {
    const bool notHeader = !DR->getTag(B).isLoopHeader();

    for (Instruction &I : *B) {
      RemapInstruction(&I, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

      // Update the phi nodes if a uniform block has any incoming blocks*
      // that are not div causing. In that case, the predicated incoming blocks
      // will never be rewired to the uniform block so we can remove the
      // incoming block from the phi node, unless 'B' is a loop header, in which
      // case its predicated preheader (if any) will be rewired to it while we
      // connect the regions).
      //
      // *NOTE a non-div-causing incoming block may or may not be a predicated
      // block. A By All block with a non-varying branch can still branch into
      // a BOSCC region, which would seem to break the SESE criteria.
      if (notHeader) {
        if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
          for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
            BasicBlock *PHIB = PHI->getIncomingBlock(i);
            if (!DR->isUniform(*PHIB) &&
                !DR->hasFlag(*PHIB,
                             BlockDivergenceFlag::eBlockHasDivergentBranch)) {
              PHI->removeIncomingValue(i--);
            }
          }
        }
      }
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::duplicateUniformLoops(Loop *L) {
  const LoopTag &LTag = DR->getTag(L);
  Loop *const uniformL = LI->AllocateLoop();

  // Either add 'uniformL' as a child of a loop or as a top level loop.
  // If it is a child loop, either add it as a child of a uniform loop if it
  // exists, otherwise as a child of a predicated loop.
  if (Loop *parentL = L->getParentLoop()) {
    auto it = LMap.find(parentL);
    if (it != LMap.end()) {
      it->second->addChildLoop(uniformL);
    } else {
      parentL->addChildLoop(uniformL);
    }
  } else {
    LI->addTopLevelLoop(uniformL);
  }

  LMap.insert({L, uniformL});

  LLVM_DEBUG(dbgs() << "Loop " << L->getName() << " has been duplicated\n");

  // Fill the loop tag.
  LoopTag *uniformLTag = &DR->getOrCreateTag(uniformL);

  // The preheader of the loop may not have been duplicated.
  BasicBlock *preheader = LTag.preheader;
  if (BasicBlock *uniformPreheader = getBlock(preheader)) {
    preheader = uniformPreheader;
  }
  uniformLTag->preheader = preheader;
  uniformLTag->header = getBlock(LTag.header);
  uniformLTag->latch = getBlock(LTag.latch);

  LLVM_DEBUG(dbgs() << "\tPreheader: " << uniformLTag->preheader->getName()
                    << "\n");
  LLVM_DEBUG(dbgs() << "\tHeader: " << uniformLTag->header->getName() << "\n");
  LLVM_DEBUG(dbgs() << "\tLatch: " << uniformLTag->latch->getName() << "\n");

  // Add all blocks to the uniform version.
  for (BasicBlock *blockL : L->blocks()) {
    if (DR->getTag(blockL).loop->loop == L) {
      BasicBlockTag &uniformBlockLTag = DR->getTag(getBlock(blockL));
      uniformL->addBasicBlockToLoop(uniformBlockLTag.BB, *LI);
      uniformBlockLTag.loop = uniformLTag;
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::createUniformRegions(
    const DenseSet<BasicBlock *> &noDuplicateBlocks) {
  auto discardRegion =
      [&noDuplicateBlocks](const UniformRegion &region) -> bool {
    // To determine if it is worth it to duplicate the uniform region, we must
    // take several elements into account:
    // - The length of the duplicated code
    // - branch probability
    // - TODO: CA-1221
    // size_t cost =
    //    std::accumulate(Region->predicatedBlocks.begin(),
    //    Region->predicatedBlocks.end(), 0,
    //                    [](int x, BasicBlock *B) { return x +
    //                    B->size(); });
    // PercentageOfAllTrue =
    // runTimeValuesOfVectorPredicateAllTrue /
    // runTimeValuesOfVectorPredicate;
    //
    // It may not be worth to duplicate the whole uniform region but still worth
    // to duplicate some of the divergent branches in it.

    if (region.predicatedBlocks.empty() /*|| cost > max*/) {
      return true;
    }

    // If the region we want to duplicate contains NoDuplicate
    // function calls, then we cannot duplicate it.
    if (std::any_of(region.predicatedBlocks.begin(),
                    region.predicatedBlocks.end(),
                    [&noDuplicateBlocks](BasicBlock *B) {
                      return noDuplicateBlocks.count(B);
                    })) {
      LLVM_DEBUG(dbgs() << "Region of " << region.entryBlock->getName()
                        << " cannot be duplicated because of "
                           "NoDuplicate instructions\n");
      return true;
    }

    // It's not worth BOSCCing if all the blocks are trivial
    if (std::all_of(region.predicatedBlocks.begin(),
                    region.predicatedBlocks.end(),
                    [](BasicBlock *B) { return isTrivialBlock(*B); })) {
      return true;
    }

    return false;
  };

  // We wish to identify Single-Entry, Single-Exit regions of the CFG
  // that contain divergence-causing branches. A SESE region is defined
  // as a subgraph of the CFG with an entry point at A and an exit point
  // at B such that:
  //   1. A dominates B
  //   2. B post-dominates A
  //   3. Any loop containing A also contains B, and vice-versa.
  //
  // The properties of the Dominance-Compact Block Indexing also happen to
  // imply SESE-compactness, so once we identify an entry point, we can
  // construct a SESE region by finding the exit block that post-dominates
  // everything in a subsequence of the DCBI starting from A.
  //
  // We had assumed initailly that any divergence-causing block will be the
  // start of a SESE region. However, certain edge cases have arisen during
  // testing that demonstrate that this is not the case. In practice, this
  // doesn't seem to matter, as long as we can fully identify the predicated
  // subset of the SESE region, so we are really working with Multiple-Entry,
  // Single-Exit regions here. This was the cause of the BOSCC Back Door bug
  // that was encountered previously (CA-2711), where the entry block of a
  // supposed SESE region did not actually dominate everything in the region,
  // which in this case was caused by an additional non-divergent code path
  // (the "back door" entry point), but it is equally possible for two
  // divergence-causing branches to enter a predicated region (CA-3194).
  //
  // a)    A*      b)    A       c)    A       d)    A      .
  //      / \           / \           / \           / \     .
  //     B   D         B*  D         B*  D*        B*  D*   .
  //    / \ / \       / \ / \       / \ / \       / \ / \   .
  //   C   F   E     C   F   E     C   F   E     C   F   E  .
  //    \  |  /       \  |  /       \  |  /       \ /   /   .
  //     \ | /         \ | /         \ | /         G   /    .
  //      \|/           \|/           \|/           \ /     .
  //       X             X             X             X      .
  //
  // Figure 1. CFGs showing SESE regions. Divergence-causing blocks are marked
  // with an asterisk. Blocks are labelled alphabetically in DCBI order.
  //
  // (1a) shows the case of a SESE region with a divergence-causing entry block.
  //
  // (1b) shows the "back door" case, where a block inside the predicated
  // sub-region has a non-divergent predecessor outside of it.
  //
  // (1c) shows a SESE region with two divergence-causing entry points into the
  // predicated sub-region. This will result in two overlapping regions.
  //
  // (1d) shows a case where the exit block of the SESE region is not the
  // immediate post-dominator of B, the first-encountered divergence causing
  // block. Therefore the two overlapping regions have different exit blocks.
  //
  // Another situation can arise (CA-3851) where the SESE region can contain
  // two completely unconnected predicated subregions. Although the DCBI is
  // SESE compact, a SESE region can still contain other, nested SESE regions.
  // Since an entry point into the predicated subregion is not necessarily the
  // SESE entry point, all predicated blocks may not be reachable from every
  // entry point. Because of these cases, it is necessary to consider each
  // divergence causing block that is not part of the predicated subregion of
  // any other divergence causing block as the entry point of their own SESE
  // regions, even though this does not strictly satisfy the SESE criteria.
  //
  // a)    A      b)       A      Figure 2.
  //      / \             / \     .
  //     B*  E*          /   D*   (2a) shows a case of two independent regions
  //    / \ / \         /   / \   sharing an exit block.
  //   C  D F  G       B*  E   F  .
  //    \ | | /       / \   \ /   (2b) shows a case where a SESE subregion will
  //     \| |/       C   \   G    appear in the middle of the DCBI of the
  //      \ /         \   \ /     subregion beginning with B. G post-dominates
  //       X           \   H      D, forming a complete nested SESE region.
  //                    \ /       .
  //                     X        .

  struct SESEInfo {
    BasicBlock *BB = nullptr;
    bool divCausing = false;
    bool predicated = false;
  };

  // Collect all the blocks in the worklist
  const auto &DCBI = DR->getBlockOrdering();
  const size_t numBlocks = DCBI.size();
  SmallVector<SESEInfo, 16> SESE;
  SESE.reserve(numBlocks);
  for (const auto &BBTag : DCBI) {
    SESE.emplace_back();
    SESE.back().BB = BBTag.BB;
  }

  // Mark all the divergence-causing blocks
  for (auto *const BB : DR->getDivCausingBlocks()) {
    SESE[DR->getTagIndex(BB)].divCausing = true;
  }

  // Create the BOSCC regions
  for (size_t i = 0; i != numBlocks;) {
    auto &info = SESE[i];
    if (!info.divCausing) {
      ++i;
      continue;
    }

    uniformRegions.emplace_back();
    auto &region = uniformRegions.back();
    const size_t entryPos = i;
    size_t exitPos = 0u;
    size_t firstPredicated = numBlocks;

    region.entryBlock = info.BB;
    region.divergentBranches.push_back(info.BB);

    SmallVector<unsigned, 16> stack;

    // If we are in a divergent loop, then the whole loop needs a uniform
    // version.
    const auto *const entryLoopTag = DR->getTag(info.BB).loop;
    if (entryLoopTag && entryLoopTag->isLoopDivergent()) {
      auto *const loop = entryLoopTag->loop;
      for (BasicBlock *loopB : loop->blocks()) {
        const size_t pos = DR->getTagIndex(loopB);
        firstPredicated = std::min(firstPredicated, pos);
        SESE[pos].predicated = true;
        region.predicatedBlocks.insert(loopB);

        if (loop->isLoopExiting(loopB)) {
          stack.push_back(pos);
        }
      }
    }

    // Traverse the CFG from the entry point, marking blocks for predication
    stack.push_back(entryPos);
    while (!stack.empty()) {
      auto *const cur = SESE[stack.pop_back_val()].BB;
      for (BasicBlock *succ : successors(cur)) {
        const size_t succPos = DR->getTagIndex(succ);

        auto *const succLoopTag = DR->getBlockTag(succPos).loop;
        if ((!succLoopTag || !succLoopTag->isLoopDivergent()) &&
            // The region 'entry' creates contains only blocks that are
            // contained in its SESE region.
            PDT->properlyDominates(succ, region.entryBlock)) {
          VECZ_ERROR_IF(exitPos != 0u && succPos != exitPos,
                        "SESE region multiple exit blocks identified");
          exitPos = succPos;
          continue;
        }

        auto &succInfo = SESE[succPos];
        if (!succInfo.predicated) {
          firstPredicated = std::min(firstPredicated, succPos);
          stack.push_back(succPos);
          region.predicatedBlocks.insert(succ);
          succInfo.predicated = true;
        }
      }
    }
    VECZ_ERROR_IF(exitPos == 0u, "SESE region exit block not identified");
    region.exitBlock = SESE[exitPos].BB;
    i = exitPos;

    // Collect any other divergent branches in the predicated region, and clear
    // the predication flags so regions can overlap.
    for (unsigned j = firstPredicated; j != exitPos; ++j) {
      auto &ji = SESE[j];
      if (ji.divCausing && j > entryPos) {
        if (ji.predicated) {
          region.divergentBranches.push_back(ji.BB);
          ji.divCausing = false;
        } else if (j < i) {
          // Found another unpredicated divergent branch between the entry
          // point and the exit point. Reset the iterator so we can process it.
          i = j;
        }
      }
      ji.predicated = false;
    }

    if (discardRegion(region)) {
      // It's not worth keeping this region.
      uniformRegions.pop_back();
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::connectBOSCCRegions() {
  LLVM_DEBUG(dbgs() << "CONNECT BOSCC REGIONS\n");

  // If we have not duplicated a loop but we have duplicated the preheader,
  // then the loop now has 2 preheaders. We thus need to blend them into one
  // single preheader.
  for (auto *const LTag : DR->getLoopOrdering()) {
    if (!LTag->isLoopDivergent() && !LMap.count(LTag->loop)) {
      BasicBlock *predicatedPreheader = LTag->preheader;
      if (BasicBlock *uniformPreheader = getBlock(predicatedPreheader)) {
        BasicBlock *header = LTag->header;

        LLVM_DEBUG(dbgs() << "Loop " << header->getName()
                          << " has two preheaders\n");

        // Create a new loop preheader that blends both the uniform and
        // predicated preheaders, to keep well formed loops (with only one
        // incoming preheader).
        BasicBlock *newPreheader = BasicBlock::Create(
            F.getContext(), predicatedPreheader->getName() + ".blend", &F,
            header);
        BranchInst::Create(header, newPreheader);

        // Set the successor of both preheaders to be the new preheader.
        auto *predicatedPreheaderT = predicatedPreheader->getTerminator();
        auto *uniformPreheaderT = uniformPreheader->getTerminator();
        VECZ_ERROR_IF(predicatedPreheaderT->getNumSuccessors() != 1,
                      "Preheader should have only one successor");
        VECZ_ERROR_IF(uniformPreheaderT->getNumSuccessors() != 1,
                      "Preheader should have only one successor");
        predicatedPreheaderT->setSuccessor(0, newPreheader);
        uniformPreheaderT->setSuccessor(0, newPreheader);

        // Update the tags.
        BasicBlockTag &newPreheaderTag = DR->getOrCreateTag(newPreheader);
        newPreheaderTag.loop = DR->getTag(predicatedPreheader).loop;
        LTag->preheader = newPreheader;

        DR->setFlag(*newPreheader, DR->getFlag(*predicatedPreheader));

        addInRegions(newPreheader, predicatedPreheader);
      }
    }
  }

  // We must make the outermost non duplicated loop's preheader target the
  // outermost duplicated uniform and predicated loop's headers. The first
  // iteration of the loop will necessarily have all lanes activated until it
  // reaches the first divergent block. Also, once the loop starts diverging,
  // there is no way to go back to a dynamically uniform loop, so there is no
  // point allowing the loop to go back and forth between its uniform and
  // predicated versions. Only going from the uniform to the predicated
  // version makes sense.
  for (const auto &pair : LMap) {
    Loop *uniformL = pair.second;
    const Loop *L = pair.first;

    if (Loop *parentL = L->getParentLoop()) {
      if (LMap.count(parentL)) {
        continue;
      }
    }

    const auto &LTag = DR->getTag(L);
    BasicBlock *preheader = LTag.preheader;
    if (!VMap.count(preheader)) {
      auto *T = preheader->getTerminator();
      VECZ_ERROR_IF(T->getNumSuccessors() != 1,
                    "Preheader has more than one successor");

      LLVM_DEBUG(dbgs() << "Non duplicated preheader " << preheader->getName()
                        << "must target uniform loop " << uniformL->getName()
                        << "\n");

      // Add a path from 'preheader' to the uniform loop header and make it
      // always branch to it. We want to keep the edge from 'preheader' to the
      // predicated loop header (even though we will never branch to it) to ease
      // some needed blendings later on.
      IRCleanup::deleteInstructionNow(T);
      BranchInst::Create(DR->getTag(uniformL).header, LTag.header,
                         ConstantInt::getTrue(F.getContext()), preheader);
    }
  }

  DenseSet<BasicBlock *> connectedBlocks;
  for (auto &region : uniformRegions) {
    // Each uniform version of div causing blocks need an entry point to the
    // predicated CFG.
    for (BasicBlock *B : region.divergentBranches) {
      if (connectedBlocks.insert(B).second) {
        if (BasicBlock *uniformB = getBlock(B)) {
          VECZ_FAIL_IF(!connectUniformRegion(region, B, uniformB));
        } else {
          VECZ_FAIL_IF(!connectUniformRegion(region, B, B));
        }
      } else {
        // No other region should have connected the entry block.
        BasicBlock *entry = region.entryBlock;
        VECZ_FAIL_IF(B == entry);
      }
    }
  }

  // If a uniform block targets a predicated block, the latter needs its
  // operands that have a uniform and predicated version blended.
  for (const auto &predicatedBTag : DR->getBlockOrdering()) {
    if (BasicBlock *uniformB = getBlock(predicatedBTag.BB)) {
      for (BasicBlock *succ : successors(uniformB)) {
        // We've found a uniform block that targets a predicated block prior
        // to connecting the regions.
        if (!DR->isUniform(*succ)) {
          LLVM_DEBUG(dbgs() << "Uniform block " << uniformB->getName()
                            << " targets predicated block " << succ->getName()
                            << "\n");
          VECZ_FAIL_IF(
              !blendConnectionPoint(succ, {predicatedBTag.BB, uniformB}));
        }
      }
    }
  }

  // Add all the uniform blocks into the worklist now they got connected.
  DT->recalculate(F);
  PDT->recalculate(F);
  VECZ_ERROR_IF(!DT->verify(), "DominatorTree incorrectly updated");
  VECZ_ERROR_IF(!PDT->verify(), "PostDominatorTree incorrectly updated");
  VECZ_FAIL_IF(!computeBlockOrdering());

  // NOTE doing the Liveness Analysis here is potentially dangerous, since we
  // have yet to fully restore SSA form (CA-3703).
  liveness = &AM.getResult<LivenessAnalysis>(F);
  RC->recalculate(F);
  VECZ_FAIL_IF(!blendFinalize());

  // Sort URVBlender in a post order so that the replaced new values don't
  // overlap with old ones.
  if (!URVB.empty()) {
    std::sort(URVB.begin(), URVB.end(),
              [this](const URVBlender::value_type &LHS,
                     const URVBlender::value_type &RHS) {
                return DR->getTagIndex(LHS.first) > DR->getTagIndex(RHS.first);
              });

    // Now that the CFG has been fully rewired and every node is correctly
    // connected, we can replace the blended values uses with their new
    // value.
    DenseSet<Instruction *> toDelete;
    for (const URVBlender::value_type &blender : URVB) {
      BasicBlock *block = blender.first;
      Value *from = blender.second.first;
      Instruction *to = blender.second.second;
      if (!isUsedOutsideDefinitionBlock(from)) {
        toDelete.insert(to);
      } else {
        VECZ_ERROR_IF(!isa<Instruction>(from),
                      "Trying to replace uses of a value");
        VECZ_FAIL_IF(
            !replaceReachableUses(*RC, cast<Instruction>(from), to, block));
      }
    }

    for (Instruction *I : toDelete) {
      IRCleanup::deleteInstructionNow(I);
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::connectUniformRegion(
    UniformRegion &region, BasicBlock *predicatedB, BasicBlock *uniformB) {
  auto replaceIncomingBlock = [](BasicBlock *B, BasicBlock *from,
                                 BasicBlock *to) {
    for (Instruction &I : *B) {
      if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
        const int fromIdx = PHI->getBasicBlockIndex(from);
        if (fromIdx != -1) {
          PHI->setIncomingBlock(fromIdx, to);
        }
      } else {
        break;
      }
    }
  };

  LLVM_DEBUG(dbgs() << "\tConnect uniform region of " << predicatedB->getName()
                    << "\n");

  ConstantInt *trueCI = ConstantInt::getTrue(F.getContext());

  auto *T = uniformB->getTerminator();

  BasicBlock *target = predicatedB->getTerminator()->getSuccessor(0);

  // 1. For each pair {taken, fallthrough} of successors of uniformB,
  //   a. 'taken' is taken if the exit mask towards that edge is full, i.e. if
  //      it contains all-true values.
  //   b. otherwise, we branch to a new block, 'boscc_indir'. If the exit mask
  //      towards 'fallthrough' is full, branch to the latter.
  //   c. Otherwise, it means the mask is not dynamically uniform, but varying,
  //      so we need to branch into the varying counterpart of the uniformregion
  //      region. The chosen block to branch to is the first successor of
  //      predicatedB.
  // 2. When a latch is divergent, we make the uniform latch target the
  //    predicated header.
  // 3. We need to feed the last computed uniform values when transitioning to
  //    the varying version.
  BasicBlock *runtimeCheckerBlock = uniformB;
  DR->setFlag(*uniformB, eBlockNeedsAllOfMask);

  // 1.
  SmallVector<BasicBlock *, 2> succs = uniformEdges[predicatedB];
  const size_t size = succs.size();
  VECZ_ERROR_IF(size == 0, "BasicBlock has no successors");
  for (size_t i = 0; i < size; ++i) {
    // Not all successors of a BOSCC entry block may be duplicated.
    if (BasicBlock *uniformSucc = getBlock(succs[i])) {
      succs[i] = uniformSucc;
    }
    LLVM_DEBUG(dbgs() << "\tSuccessor " << i << ": " << succs[i]->getName()
                      << "\n");
  }

  for (size_t i = 0; i + 1 < size; ++i) {
    BasicBlock *succ = succs[i];

    BasicBlock *BOSCCIndir = BasicBlock::Create(
        uniformB->getContext(), uniformB->getName() + ".boscc_indir", &F,
        succ->getNextNode());

    region.uniformBlocks.insert(BOSCCIndir);

    BasicBlockTag &BOSCCIndirTag = DR->getOrCreateTag(BOSCCIndir);
    DR->setFlag(*BOSCCIndir, static_cast<BlockDivergenceFlag>(
                                 eBlockNeedsAllOfMask | eBlockIsUniform));
    BOSCCIndirTag.loop = DR->getTag(runtimeCheckerBlock).loop;
    if (BOSCCIndirTag.loop) {
      BOSCCIndirTag.loop->loop->addBasicBlockToLoop(BOSCCIndir, *LI);
    }

    ICmpInst *cond = new ICmpInst(
        *runtimeCheckerBlock, CmpInst::ICMP_EQ,
        PassState.getMaskInfo(uniformB).exitMasks.lookup(succ), trueCI);
    BranchInst::Create(succ, BOSCCIndir, cond, runtimeCheckerBlock);

    if (i > 0) {
      // Update the incoming block of the phi nodes in 'succ' from 'uniformB'
      // to 'runtimeCheckerBlock'.
      replaceIncomingBlock(succ, uniformB, runtimeCheckerBlock);
    }

    runtimeCheckerBlock = BOSCCIndir;
  }

  BasicBlock *succ = succs[size - 1];
  ICmpInst *cond = new ICmpInst(
      *runtimeCheckerBlock, CmpInst::ICMP_EQ,
      PassState.getMaskInfo(uniformB).exitMasks.lookup(succ), trueCI);

  BasicBlock *connectionPoint = target;

  const auto *const LTag = DR->getTag(predicatedB).loop;
  const bool needsStore = LTag && LMap.count(LTag->loop);
  if (needsStore) {
    // 'store' is a block that will contain all the uniform versions of the
    // live in instructions of the predicated target.
    BasicBlock *store = BasicBlock::Create(
        target->getContext(), uniformB->getName() + ".boscc_store", &F,
        runtimeCheckerBlock->getNextNode());

    region.uniformBlocks.insert(store);

    BasicBlockTag &storeTag = DR->getOrCreateTag(store);
    DR->setFlag(*store, eBlockIsUniform);

    // 2.
    auto *const uniformLTag = DR->getTag(uniformB).loop;
    const bool isLoopLatch = uniformLTag && (uniformLTag->latch == uniformB);
    if (isLoopLatch) {
      BasicBlock *header = LTag->header;
      PHINode *entryMask =
          cast<PHINode>(PassState.getMaskInfo(header).entryMask);
      Value *latchMask =
          PassState.getMaskInfo(uniformB).exitMasks.lookup(uniformLTag->header);
      VECZ_ERROR_IF(!latchMask, "Exit mask does not exist");
      entryMask->addIncoming(latchMask, store);
      connectionPoint = header;

      if (succ == uniformLTag->header) {
        uniformLTag->latch = runtimeCheckerBlock;
      }
    }

    BranchInst::Create(connectionPoint, store);

    // 'store' belongs in the first outer loop non duplicated.
    Loop *parentLoop = LTag->loop->getParentLoop();
    while (parentLoop && LMap.count(parentLoop)) {
      parentLoop = parentLoop->getParentLoop();
    }
    if (parentLoop) {
      storeTag.loop = &DR->getTag(parentLoop);
      parentLoop->addBasicBlockToLoop(store, *LI);
    }

    target = store;
  }

  // 1.c. 'uniformB' has a new runtime check, we can remove its old one.
  IRCleanup::deleteInstructionNow(T);
  BranchInst::Create(succ, target, cond, runtimeCheckerBlock);

  // Update the incoming block of the new successors of 'runTimeCheckerBlock'.
  replaceIncomingBlock(succ, uniformB, runtimeCheckerBlock);

  if (uniformB == predicatedB) {
    replaceIncomingBlock(connectionPoint, predicatedB, runtimeCheckerBlock);
  } else {
    // 3.
    VECZ_FAIL_IF(!blendConnectionPoint(
        connectionPoint,
        {predicatedB, needsStore ? target : runtimeCheckerBlock}));

    if (needsStore) {
      region.storeBlocks.emplace_back();
      auto &sb = region.storeBlocks.back();
      sb.connectionPoint = connectionPoint;
      sb.target = target;
      sb.runtimeCheckerBlock = runtimeCheckerBlock;
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::blendConnectionPoint(
    BasicBlock *CP, const std::pair<BasicBlock *, BasicBlock *> &incoming) {
  const auto *const CPLTag = DR->getTag(CP).loop;
  for (auto &region : uniformRegions) {
    // Create blend instructions at each blend point following 'CP'.
    if (region.contains(CP) || (CP == region.exitBlock) ||
        (CP == region.entryBlock)) {
      // Compute all the blend points that will need to have blend instructions
      // because of 'CP'. These blocks are all the blocks that have more than
      // one predecessor, that belong to the same region as 'CP', and that
      // succeed it.
      if (!region.blendPoints.count(CP)) {
        // The first blend point impacted by 'CP' is 'CP' itself.
        region.blendPoints.insert({CP, {CP}});

        DenseSet<BasicBlock *> visited{CP};
        std::queue<BasicBlock *> queue;
        queue.push(CP);
        while (!queue.empty()) {
          BasicBlock *cur = queue.front();
          queue.pop();
          // The region exit block is the delimiter of the region.
          if (cur == region.exitBlock) {
            continue;
          }
          for (BasicBlock *succ : successors(cur)) {
            if (visited.insert(succ).second) {
              queue.push(succ);
              if (std::distance(pred_begin(succ), pred_end(succ)) > 1) {
                // Nested loops are dominated.
                if (CPLTag == DR->getTag(succ).loop ||
                    (CPLTag && !CPLTag->loop->contains(succ))) {
                  region.blendPoints[CP].push_back(succ);
                }
              }
            }
          }
        }
      }

      region.connections.push_back(UniformRegion::ConnectionInfo{CP, incoming});
    }
  }
  return true;
}

bool ControlFlowConversionState::BOSCCGadget::blendFinalize() {
  for (auto &region : uniformRegions) {
    for (const auto &connection : region.connections) {
      BasicBlock *CP = connection.connectionPoint;
      auto &incoming = connection.incoming;

      // Create blend instructions at each blend point following 'CP'.
      for (BasicBlock *blendPoint : region.blendPoints[CP]) {
        LLVM_DEBUG(dbgs() << "BLEND CONNECTION POINT " << blendPoint->getName()
                          << "\n");

        for (Instruction &I : *blendPoint) {
          if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
            // Only add 'incoming' for 'CP' because for the other blend points
            // we don't actually add a new edge.
            if (blendPoint != CP ||
                PHI->getBasicBlockIndex(incoming.second) != -1) {
              continue;
            }

            unsigned idx = 0;
            for (; idx < PHI->getNumIncomingValues(); ++idx) {
              // If one incoming block of the phi node is the predicated version
              // of the new, uniform, incoming block, use its uniform incoming
              // value version if it exists.
              if (PHI->getIncomingBlock(idx) == incoming.first) {
                if (Value *V = getUniformV(PHI->getIncomingValue(idx))) {
                  if (Instruction *VI = dyn_cast<Instruction>(V)) {
                    if (RC->isReachable(VI->getParent(), incoming.second)) {
                      PHI->addIncoming(VI, incoming.second);
                      break;
                    }
                  }
                }
              }
            }
            if (idx == PHI->getNumIncomingValues()) {
              PHI->addIncoming(getDefaultValue(PHI->getType()),
                               incoming.second);
            }
            LLVM_DEBUG(
                dbgs()
                << "PHINode " << PHI->getName() << ": Add incoming value "
                << PHI->getIncomingValueForBlock(incoming.second)->getName()
                << " from " << incoming.second->getName() << " in "
                << blendPoint->getName() << "\n");
          } else {
            break;
          }
        }
      }
    }
    region.connections.clear();
  }

  DenseSet<BasicBlock *> blendBlocks;
  for (const auto &region : uniformRegions) {
    for (auto &CP : region.blendPoints) {
      for (BasicBlock *blendPoint : CP.second) {
        blendBlocks.insert(blendPoint);
      }
    }
  }

  for (const auto &tag : DR->getBlockOrdering()) {
    BasicBlock *blendPoint = tag.BB;
    if (blendBlocks.count(blendPoint) == 0) {
      continue;
    }

    DenseSet<Value *> blendedValues;
    for (Instruction &I : *blendPoint) {
      if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
        if (PHI->getName().contains(".boscc_blend")) {
          for (Value *v : PHI->incoming_values()) {
            blendedValues.insert(v);
          }
        }
      } else {
        break;
      }
    }

    for (auto *liveInVal : liveness->getBlockInfo(blendPoint).LiveIn) {
      if (blendedValues.count(liveInVal)) {
        continue;
      }

      auto *liveIn = dyn_cast<Instruction>(liveInVal);
      if (!liveIn) {
        continue;
      }

      BasicBlock *src = liveIn->getParent();

      // Nothing to be done if the definition block has no uniform
      // equivalent.
      BasicBlock *uniformSrc = getBlock(src);
      if (!uniformSrc) {
        continue;
      }

      // Nothing to be done if the instruction:
      // - dominates the connection point,
      // - cannot reach 'CP'.
      if (DT->dominates(src, blendPoint)) {
        continue;
      }

      if (!RC->isReachable(src, blendPoint)) {
        continue;
      }

      Value *uniformLiveIn = getDefaultValue(liveIn->getType());
      if (Value *V = getUniformV(liveIn)) {
        uniformLiveIn = V;
      }

      LLVM_DEBUG(dbgs() << "Blend live in " << liveIn->getName() << " in "
                        << blendPoint->getName() << "\n");

      PHINode *blend = PHINode::Create(liveIn->getType(), 2,
                                       liveIn->getName() + ".boscc_blend",
                                       &blendPoint->front());
      bool replaceUniform = false;
      bool replacePredicate = false;
      // For each predecessor, if it can reach the instruction, set the
      // latter as the incoming value, otherwise set a default value.
      for (BasicBlock *pred : predecessors(blendPoint)) {
        if (DR->isUniform(*pred)) {
          Instruction *uniformLiveInI = dyn_cast<Instruction>(uniformLiveIn);
          if (uniformLiveInI &&
              !RC->isReachable(uniformLiveInI->getParent(), pred)) {
            blend->addIncoming(getDefaultValue(uniformLiveInI->getType()),
                               pred);
          } else {
            replaceUniform = true;
            blend->addIncoming(uniformLiveIn, pred);
          }
        } else if (DR->getTag(pred).isLoopBackEdge(blendPoint)) {
          blend->addIncoming(blend, pred);
        } else {
          if (!RC->isReachable(liveIn->getParent(), pred)) {
            blend->addIncoming(getDefaultValue(liveIn->getType()), pred);
          } else {
            replacePredicate = true;
            blend->addIncoming(liveIn, pred);
          }
        }
        LLVM_DEBUG(dbgs() << "\tAdd incoming value "
                          << blend->getIncomingValueForBlock(pred)->getName()
                          << " from " << pred->getName() << "\n");
      }

      // If we have blended 'liveIn' in 'CP', update the uses.
      if (replacePredicate) {
        URVB.push_back({blendPoint, {liveIn, blend}});
        addReference(blend, liveIn);
      }
      // If we have blended 'uniformLiveIn' in 'CP', update the uses.
      if (replaceUniform && isa<Instruction>(uniformLiveIn)) {
        URVB.push_back({blendPoint, {uniformLiveIn, blend}});
      }

      // Update the blend instructions in the loop header, if any.
      VECZ_FAIL_IF(
          !updateLoopBlendValues(DR->getTag(blendPoint).loop, liveIn, blend));
      blendedValues.insert(liveIn);
    }
  }

  for (const auto &region : uniformRegions) {
    for (auto &sb : region.storeBlocks) {
      BasicBlock *connectionPoint = sb.connectionPoint;
      BasicBlock *target = sb.target;
      BasicBlock *runtimeCheckerBlock = sb.runtimeCheckerBlock;

      // Create a bunch of lcssa instructions into 'store' so that the repair
      // SSA doesn't have to look for the instructions inside the uniform loop.
      for (Instruction &I : *connectionPoint) {
        if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
          const int idx = PHI->getBasicBlockIndex(target);
          VECZ_ERROR_IF(idx == -1,
                        "Connection point PHIs must have incoming "
                        "block from the target");
          if (Instruction *incoming =
                  dyn_cast<Instruction>(PHI->getIncomingValue(idx))) {
            LLVM_DEBUG(dbgs()
                       << "Create live-in lcssa of " << incoming->getName()
                       << " in " << target->getName() << "\n");

            PHINode *blend = PHINode::Create(
                incoming->getType(), 1, incoming->getName() + ".boscc_lcssa",
                &target->front());
            blend->addIncoming(incoming, runtimeCheckerBlock);
            PHI->setIncomingValue(idx, blend);
          }
        } else {
          break;
        }
      }
    }
  }
  return true;
}

BasicBlock *ControlFlowConversionState::BOSCCGadget::getBlock(BasicBlock *B) {
  auto BUniform = VMap.find(B);
  if (BUniform != VMap.end()) {
    return cast<BasicBlock>(BUniform->second);
  }
  return nullptr;
}

Loop *ControlFlowConversionState::BOSCCGadget::getLoop(Loop *L) {
  auto LUniform = LMap.find(L);
  if (LUniform != LMap.end()) {
    return LUniform->second;
  }
  return nullptr;
}

void ControlFlowConversionState::BOSCCGadget::getUnduplicatedEntryBlocks(
    SmallVectorImpl<BasicBlock *> &blocks) const {
  for (const auto &region : uniformRegions) {
    if (VMap.count(region.entryBlock) == 0) {
      blocks.push_back(region.entryBlock);
    }
  }
}

void ControlFlowConversionState::BOSCCGadget::createReference(
    Value *pred, Value *uni, bool needsMapping) {
  if (!pred || !uni) {
    return;
  }
  auto predIt = VMap.find(pred);
  if (predIt != VMap.end()) {
    predIt->second = uni;
  } else {
    VMap.insert({pred, uni});
  }

  if (needsMapping) {
    if (Instruction *uniI = dyn_cast<Instruction>(uni)) {
      RemapInstruction(uniI, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    }
  }
}

void ControlFlowConversionState::BOSCCGadget::addReference(Value *pred,
                                                           Value *old) {
  auto uniformOldIt = VMap.find(old);
  if (uniformOldIt != VMap.end()) {
    VMap.insert({pred, uniformOldIt->second});
  }
}

void ControlFlowConversionState::BOSCCGadget::addInRegions(BasicBlock *newB,
                                                           BasicBlock *refB) {
  for (auto &region : uniformRegions) {
    if (region.contains(refB)) {
      if (region.predicatedBlocks.insert(newB).second) {
        LLVM_DEBUG(dbgs() << "BasicBlock " << newB->getName()
                          << " added to BOSCC region: "
                          << region.entryBlock->getName() << "\n");
      }
    }
  }
}

Value *ControlFlowConversionState::BOSCCGadget::getUniformV(
    Value *predicatedV) {
  auto uniformVIt = VMap.find(predicatedV);
  if (uniformVIt != VMap.end()) {
    return uniformVIt->second;
  }
  return nullptr;
}

void ControlFlowConversionState::BOSCCGadget::updateValue(Value *from,
                                                          Value *to) {
  auto fromIt = VMap.find(from);
  if (fromIt != VMap.end()) {
    Value *fromUniform = fromIt->second;
    VMap.erase(from);
    VMap.insert({to, fromUniform});
  }
}

bool ControlFlowConversionState::BOSCCGadget::linkMasks() {
  for (const auto &BTag : DR->getBlockOrdering()) {
    auto *const BB = BTag.BB;
    if (auto *const uniformB = getBlock(BB)) {
      // Both sets of masks had better exist by this point.
      auto &masks = PassState.getMaskInfo(BB);
      auto &masksUniform = PassState.getMaskInfo(uniformB);
      createReference(masks.entryMask, masksUniform.entryMask);

      for (auto *const succ : successors(BB)) {
        auto *const uniformSucc = getBlock(succ);
        auto *const target = uniformSucc ? uniformSucc : succ;
        createReference(masks.exitMasks.lookup(succ),
                        masksUniform.exitMasks.lookup(target));
      }
    }
  }
  return true;
}

bool ControlFlowConversionState::BOSCCGadget::updateLoopBlendValues(
    LoopTag *LTag, Instruction *from, Instruction *to) {
  auto createLatchIncoming = [&from, &LTag, this] {
    auto *ret =
        PHINode::Create(from->getType(), 2, from->getName() + ".boscc_blend",
                        &LTag->latch->front());
    Value *uniform = getUniformV(from);
    Value *default_val = getDefaultValue(from->getType());
    for (BasicBlock *pred : predecessors(LTag->latch)) {
      Value *incoming = default_val;
      if (RC->isReachable(from->getParent(), pred)) {
        incoming = from;
      } else if (uniform) {
        Instruction *uinst = dyn_cast<Instruction>(uniform);
        if (!uinst || RC->isReachable(uinst->getParent(), pred)) {
          incoming = uniform;
        }
      }
      ret->addIncoming(incoming, pred);
    }
    URVB.push_back({LTag->latch, {from, ret}});
    addReference(ret, from);
    return ret;
  };

  while (LTag) {
    PHINode *latchIncoming = nullptr;
    // Try looking for an existing `boscc_blend` value for `from` to avoid
    // creating a new one in the latch.
    for (Instruction &latchI : *LTag->latch) {
      if (PHINode *PHI = dyn_cast<PHINode>(&latchI)) {
        if (PHI->getName().contains(".boscc_blend")) {
          for (Value *incomingValue : PHI->incoming_values()) {
            if (incomingValue == from) {
              latchIncoming = PHI;
              break;
            }
          }
          if (latchIncoming) {
            break;
          }
        }
      } else {
        break;
      }
    }
    // Update all uses of `from` in the header with the blended value from the
    // latch. Since the CFG is final now, this should cover everything.
    for (Instruction &headerI : *LTag->header) {
      if (PHINode *PHI = dyn_cast<PHINode>(&headerI)) {
        const int latchIdx = PHI->getBasicBlockIndex(LTag->latch);
        VECZ_ERROR_IF(latchIdx == -1,
                      "Header has no incoming value from the latch");
        if ((PHI == to) || (PHI->getIncomingValue(latchIdx) == from)) {
          if (!latchIncoming) {
            latchIncoming = createLatchIncoming();
          }
          PHI->setIncomingValue(latchIdx, latchIncoming);
        }
      } else {
        break;
      }
    }

    if (Loop *L = LTag->loop->getParentLoop()) {
      LTag = &DR->getTag(L);
    } else {
      break;
    }
  }

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::computeBlockOrdering() {
  // Create a map from entry blocks to their uniform regions
  DenseMap<BasicBlock *, const UniformRegion *> entryMap;
  unsigned maxUBlocks = 0;
  for (const auto &region : uniformRegions) {
    if (!region.uniformBlocks.empty()) {
      entryMap[region.entryBlock] = &region;
    }
    maxUBlocks = std::max(maxUBlocks, region.uniformBlocks.size());
  }

  // Gather the blocks outside of the uniform regions according to the already
  // computed order, leaving gaps for the uniform regions to fill in.
  // Note that uniform region blocks do not appear in the block ordering yet.
  // Also note that we can't use pointers to BasicBlockTags here since
  // `PassState.computeBlockOrdering()` re-orders the tags vector.
  SmallVector<BasicBlock *, 16> filtered;
  for (const auto &tag : DR->getBlockOrdering()) {
    filtered.push_back(tag.BB);
    const auto found = entryMap.find(tag.BB);
    if (found != entryMap.end()) {
      const auto *const region = found->second;
      filtered.resize(filtered.size() + region->uniformBlocks.size());
    }
  }

  // Recompute the ordering over the uniform regions
  VECZ_FAIL_IF(!PassState.computeBlockOrdering());

  // Filter by region and fill in the gaps
  SmallVector<size_t, 16> uniformBlocks;
  uniformBlocks.reserve(maxUBlocks);
  for (auto it = filtered.begin(), ie = filtered.end(); it != ie;) {
    auto *const BB = *it;

    const auto found = entryMap.find(BB);
    if (found != entryMap.end()) {
      // If the entry block of the region is NOT duplicated, add the uniform
      // blocks after it.
      const bool entryDupe = getBlock(BB);
      if (!entryDupe) {
        ++it;
      }

      // Gather the indices of the uniform blocks and sort them.
      const auto &region = *found->second;
      uniformBlocks.clear();
      for (auto *const uBB : region.uniformBlocks) {
        uniformBlocks.push_back(DR->getTagIndex(uBB));
      }
      std::sort(uniformBlocks.begin(), uniformBlocks.end());

      // Insert the uniform blocks into the gap.
      for (const auto uBBi : uniformBlocks) {
        (*it++) = DR->getBlockTag(uBBi).BB;
      }

      // If the entry block of the region IS duplicated, add it after the
      // uniform blocks.
      if (entryDupe) {
        (*it++) = BB;
      }
    } else {
      ++it;
    }
  }

  uint32_t pos = 0;
  for (auto *const BB : filtered) {
    DR->getTag(BB).pos = pos++;
  }
  DR->reorderTags(filtered.size());

  return true;
}

bool ControlFlowConversionState::BOSCCGadget::cleanUp() {
  // BOSCC can create a lot of PHI nodes that are not really necessary.
  // LCSSA PHI nodes (in Store Blocks) are only required as an intermediate
  // state and are trivially redundant, and sometimes blends are created that
  // blend the same two values together. Also, sometimes values are blended
  // even though they have no further uses and can be removed as dead code.

  const RPOT rpot(&F);
  std::vector<PHINode *> blends;
  for (auto *BB : rpot) {
    for (auto I = BB->begin(); I != BB->end();) {
      auto *PHI = dyn_cast<PHINode>(&*(I++));
      if (!PHI) {
        break;
      }
      if (!PHI->getName().contains(".boscc_")) {
        continue;
      }

      if (auto *V = PHI->hasConstantValue()) {
        PHI->replaceAllUsesWith(V);
        IRCleanup::deleteInstructionNow(PHI);
      } else {
        blends.push_back(PHI);
      }
    }
  }

  while (!blends.empty()) {
    PHINode *PHI = blends.back();
    if (PHI->use_empty()) {
      IRCleanup::deleteInstructionNow(PHI);
    }
    blends.pop_back();
  }

  return true;
}
