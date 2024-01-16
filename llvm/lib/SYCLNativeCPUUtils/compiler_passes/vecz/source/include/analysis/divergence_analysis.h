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

/// @file
///
/// @brief Divergence analysis.

#ifndef VECZ_ANALYSIS_DIVERGENCE_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_DIVERGENCE_ANALYSIS_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/PassManager.h>

#include <vector>

namespace llvm {
class BasicBlock;
class Loop;
}  // namespace llvm

namespace vecz {
struct BasicBlockTag;
struct LoopTag;

/// @brief Analysis flags that can be attached to LLVM basic blocks.
enum BlockDivergenceFlag {
  /// @brief Flag value where no flag is set.
  eBlockHasNoFlag = 0,
  /// @brief True if the block has a divergent branch (different paths might be
  /// taken by different work items.
  eBlockHasDivergentBranch = (1 << 0),
  /// @brief True if the block has no divergent branch but has all its
  /// successors divergent.
  eBlockHasDivergentBranchFake = (1 << 1),
  /// @brief True if the block belongs in a diverged path.
  eBlockIsDivergent = (1 << 2),
  /// @brief True if the block is an introduced divergent conditional loop exit.
  /// The operation is performed during the transformation of a divergent loop.
  eBlockIsVirtualDivergentLoopExit = (1 << 3),
  /// @brief True if the block is a join point of a divergent branch.
  eBlockIsBlend = (1 << 4),
  /// @brief True if no divergence is present when reaching the block.
  eBlockIsByAll = (1 << 5),
  /// @brief True if the block is uniform (duplicated version of a predicated
  /// block from BOSCC).
  eBlockIsUniform = (1 << 6),
  /// @brief True if the block needs an all-of mask.
  eBlockNeedsAllOfMask = (1 << 7)
};

/// @brief Analysis flags that can be attached to LLVM loops.
enum LoopDivergenceFlag {
  /// @brief Flag value where no flag is set.
  eLoopNoFlag = 0,
  /// @brief Whether or not the loop may diverge because of a diverging block.
  eLoopIsDivergent = (1 << 0)
};

/// @brief Type that maps basic blocks to tags.
using DenseBBMap = llvm::DenseMap<const llvm::BasicBlock *, size_t>;
/// @brief Type that maps loops to tags.
using DenseLoopMap =
    llvm::DenseMap<const llvm::Loop *, std::unique_ptr<LoopTag>>;
/// @brief Type that maps loop live values and their associated state from the
///        previous loop iteration.
using DenseLoopResultPHIsMap =
    llvm::SmallDenseMap<llvm::Value *, llvm::PHINode *, 32>;
/// @brief Type that maps loop live values and updated value.
using DenseLoopResultUpdatesMap =
    llvm::SmallDenseMap<llvm::Value *, llvm::SelectInst *, 32>;

class DivergenceResult;

/// @brief Queue that orders blocks by their DCBI (smallest first).
struct BlockQueue {
  using index_type = uint32_t;
  using index_list = std::vector<index_type>;

  const DivergenceResult &DR;

  /// @brief The DCBI indices of the blocks in the queue, in min-heap order.
  /// Since we can easily retrieve the BasicBlockTag from the DCBI ordered
  /// `blockOrdering` vector, and since the queue priority is entirly based on
  /// the index, it is sufficient to store only the indices to perform the
  /// queue operations.
  index_list indices;

  /// @brief Constructs an empty BlockQueue
  BlockQueue(const DivergenceResult &dr) : DR(dr){};

  /// @brief Constructs a BlockQueue from a set of blocks.
  BlockQueue(const DivergenceResult &dr,
             const llvm::DenseSet<llvm::BasicBlock *> &blocks);

  /// @brief Returns the number of blocks in the queue.
  size_t size() const { return indices.size(); }

  /// @brief Returns whether the queue is empty.
  bool empty() const { return indices.empty(); }

  /// @brief Pushes a block on the queue by its DCBI index.
  void push(size_t index);

  /// @brief Pushes a block on the queue by pointer.
  /// Prefer `push(size_t)` if the tag index is available.
  void push(const llvm::BasicBlock *bb);

  /// @brief Pops a block from the queue and returns it.
  const BasicBlockTag &pop();

  /// @brief Const iterator to beginning of index list, for inspection.
  index_list::const_iterator begin() const { return indices.begin(); }

  /// @brief Const iterator to end of index list, for inspection.
  index_list::const_iterator end() const { return indices.end(); }
};

/// @brief Describes a loop contained in the function to vectorize.
struct LoopTag {
  /// @brief Compiler loop info.
  llvm::Loop *loop = nullptr;
  /// @brief Loop entering point.
  llvm::BasicBlock *preheader = nullptr;
  /// @brief Loop entry point.
  llvm::BasicBlock *header = nullptr;
  /// @brief Single block that jumps back to the loop header.
  llvm::BasicBlock *latch = nullptr;
  /// @brief Loop live values on the loop.
  llvm::SmallPtrSet<llvm::Value *, 32> loopLiveValues;
  /// @brief Map between loop live values and their associated state from the
  ///        previous loop iteration.
  DenseLoopResultPHIsMap loopResultPrevs;
  /// @brief Map between loop live values and their updated value.
  DenseLoopResultUpdatesMap loopResultUpdates;
  /// @brief Loop exit that has been chosen during partial linearization.
  llvm::BasicBlock *pureExit = nullptr;

  LoopDivergenceFlag divergenceFlag = LoopDivergenceFlag::eLoopNoFlag;

  bool isLoopDivergent() const {
    return divergenceFlag & LoopDivergenceFlag::eLoopIsDivergent;
  }
};

/// @brief Describes a basic block contained in the function to vectorize.
struct BasicBlockTag {
  /// @brief Compiler basic block object.
  llvm::BasicBlock *BB = nullptr;
  /// @brief Inner most loop this block belongs to, if any.
  LoopTag *loop = nullptr;
  /// @brief Outermost loop left by this block.
  LoopTag *outermostExitedLoop = nullptr;

  /// @brief Unique sorted block index.
  uint32_t pos = ~0u;

  /// @brief Create a new basic block tag.
  BasicBlockTag() = default;
  /// @brief Deleted address-of operator
  BasicBlockTag *operator&() = delete;
  /// @brief Deleted const address-of operator
  const BasicBlockTag *operator&() const = delete;

  BlockDivergenceFlag divergenceFlag = BlockDivergenceFlag::eBlockHasNoFlag;

  /// @brief Convenience function for finding the varying property of the branch
  /// without having to query the Uniform Value Analysis
  bool hasVaryingBranch() const {
    return divergenceFlag & BlockDivergenceFlag::eBlockHasDivergentBranch;
  }

  /// @brief Determine whether there is a backedge from this tag's basic block
  /// to the target basic block.
  ///
  /// @param[in] toBB Potential target for the backedge.
  ///
  /// @return true if there is a backedge, false otherwise.
  bool isLoopBackEdge(llvm::BasicBlock *toBB) const {
    return loop && (loop->latch == BB) && (loop->header == toBB);
  }

  /// @brief Determine whether this block is the header of its loop (if any).
  /// @return true iff the block is the loop header for its loop
  bool isLoopHeader() const { return loop && loop->header == BB; }
};

/// @brief Divergent blocks whose PHI nodes may vary.
using DivergenceInfo = llvm::DenseSet<llvm::BasicBlock *>;

/// @brief Holds the result of Divergence Analysis for a given function.
class DivergenceResult {
 public:
  /// @brief Create a new DA result for the given unit.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  DivergenceResult(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Generate a block ordering.
  ///
  /// This is based on a dominance-compact block indexing (DCBI) where we
  /// topologically order blocks that belong to the same dominator tree.
  ///
  /// @returns true if no errors occurred.
  bool computeBlockOrdering(llvm::DominatorTree &DT);

  /// @brief Reorders the tags in the tags vector according to their DBCI
  /// indices.
  /// @param[in] n the number of tags in the DCBI
  void reorderTags(size_t n);

  /// @brief Generate a loop ordering.
  ///
  /// This populates the `loopOrdering` vector with loop tags sorted by depth.
  ///
  /// @returns true if no errors occurred.
  bool computeLoopOrdering();

  /// @brief Gets a BasicBlockTag by its DCBI index
  /// @param[in] index the DCBI index
  /// @returns reference to the BasicBlockTag
  const BasicBlockTag &getBlockTag(size_t index) const {
    return basicBlockTags[index];
  }

  /// @brief Gets the DCBI ordered range of BasicBlockTags.
  llvm::ArrayRef<BasicBlockTag> getBlockOrdering() const {
    return llvm::ArrayRef<BasicBlockTag>(basicBlockTags.data(),
                                         numOrderedBlocks);
  }

  llvm::ArrayRef<LoopTag *> getLoopOrdering() { return loopOrdering; }

  size_t getTagIndex(const llvm::BasicBlock *BB) const;

  /// @brief Retrieve a tag for the given basic block.
  ///
  /// @param[in] BB Basic block to retrieve a tag for.
  ///
  /// @return Basic block tag.
  BasicBlockTag &getTag(const llvm::BasicBlock *BB) {
    return basicBlockTags[getTagIndex(BB)];
  }

  const BasicBlockTag &getTag(const llvm::BasicBlock *BB) const {
    return basicBlockTags[getTagIndex(BB)];
  }

  /// @brief Retrieve or create a tag for the given basic block.
  ///
  /// @param[in] BB Basic block to retrieve or create a tag for.
  ///
  /// @return Basic block tag.
  BasicBlockTag &getOrCreateTag(llvm::BasicBlock *BB);

  /// @brief Try to retrieve a tag for the given loop.
  ///
  /// @param[in] L Loop to retrieve a tag for.
  ///
  /// @return Loop tag.
  LoopTag &getTag(const llvm::Loop *L) const;

  /// @brief Retrieve or create a tag for the given loop.
  ///
  /// @param[in] L Loop to retrieve a tag for.
  ///
  /// @return Loop tag.
  LoopTag &getOrCreateTag(llvm::Loop *L);

  /// @brief Determine whether the tag contains the given flags or not.
  ///
  /// @param[in] BB Basic block whose flag we check.
  /// @param[in] F Flags to test.
  ///
  /// @return true if the tag contains all the given flags, false otherwise.
  bool hasFlag(const llvm::BasicBlock &BB, BlockDivergenceFlag F) const;
  /// @brief Get the given flags for the tag.
  ///
  /// @param[in] BB Basic block whose flag we want to get.
  BlockDivergenceFlag getFlag(const llvm::BasicBlock &BB) const;
  /// @brief Set the given flags for the tag.
  ///
  /// @param[in] BB Basic block whose flag we set.
  /// @param[in] F Flags to set for the tag.
  void setFlag(const llvm::BasicBlock &BB, BlockDivergenceFlag F);
  /// @brief Clear the given flags for the tag.
  ///
  /// @param[in] BB Basic block whose flag we clear.
  /// @param[in] F Flags to clear for the tag.
  void clearFlag(const llvm::BasicBlock &BB, BlockDivergenceFlag F);
  /// @brief Check whether the basic block contains a div causing flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is div causing, false otherwise.
  bool isDivCausing(const llvm::BasicBlock &BB) const;
  /// @brief Check whether the basic block contains a divergent flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is divergent, false otherwise.
  bool isDivergent(const llvm::BasicBlock &BB) const;
  /// @brief Check whether the basic block contains an optional flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is optional, false otherwise.
  bool isOptional(const llvm::BasicBlock &BB) const;
  /// @brief Check whether the basic block contains a by_all flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is by_all, false otherwise.
  bool isByAll(const llvm::BasicBlock &BB) const;
  /// @brief Check whether the basic block contains a blend flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is blend, false otherwise.
  bool isBlend(const llvm::BasicBlock &BB) const;
  /// @brief Check whether the basic block contains a uniform flag.
  ///
  /// @param[in] BB Basic block whose flag we check.
  ///
  /// @return true if the tag is uniform, false otherwise.
  bool isUniform(const llvm::BasicBlock &BB) const;

  /// @brief Determine whether the tag contains the given flags or not.
  ///
  /// @param[in] L Loop whose flag we check.
  /// @param[in] F Flags to test.
  ///
  /// @return true if the tag contains all the given flags, false otherwise.
  bool hasFlag(const llvm::Loop &L, LoopDivergenceFlag F) const;
  /// @brief Get the given flags for the tag.
  ///
  /// @param[in] L Loop whose flag we want to get.
  LoopDivergenceFlag getFlag(const llvm::Loop &L) const;
  /// @brief Set the given flags for the tag.
  ///
  /// @param[in] L Loop whose flag we set.
  /// @param[in] F Flags to set for the tag.
  void setFlag(const llvm::Loop &L, LoopDivergenceFlag F);
  /// @brief Clear the given flags for the tag.
  ///
  /// @param[in] L Loop whose flag we clear.
  /// @param[in] F Flags to clear for the tag.
  void clearFlag(const llvm::Loop &L, LoopDivergenceFlag F);

  /// @brief Check if a block Src can reach a block Dst, either within the same
  ///        SESE region, or outside too.
  /// @param[in] src Source node.
  /// @param[in] dst Destination node.
  /// @param[in] allowLatch Whether reachability is computed with latches or
  /// not.
  /// @return Whether or not dst is reachable from src.
  bool isReachable(llvm::BasicBlock *src, llvm::BasicBlock *dst,
                   bool allowLatch = false) const;

  /// @brief List of blocks having a divergent branch.
  const std::vector<llvm::BasicBlock *> &getDivCausingBlocks() const {
    return divCausingBlocks;
  }

 private:
  friend class DivergenceAnalysis;

  /// @brief Mark a block div causing and mark blocks that are control dependent
  ///        to be divergent
  /// @param[in] BB Div causing block.
  /// @param[in,out] DI Divergence information of the function.
  /// @param[in,out] PDT PostDominatorTree of the function.
  void markDivCausing(llvm::BasicBlock &BB, DivergenceInfo &DI,
                      llvm::PostDominatorTree &PDT);
  /// @brief Mark divergent blocks in a loop (loop exits and latch) that are
  ///        control dependent of a divergent branch.
  /// @param[in] BB Div causing block.
  /// @param[in] L Loop that BB diverges.
  /// @param[in,out] DI Divergence information of the function.
  void markDivLoopDivBlocks(llvm::BasicBlock &BB, llvm::Loop &L,
                            DivergenceInfo &DI);
  /// @brief Mark a block to be divergent.
  /// @param[in] BB Block to mark.
  void markDivergent(const llvm::BasicBlock &BB);
  /// @brief Mark a loop to be divergent.
  /// @param[in] L Loop to mark.
  void markDivergent(const llvm::Loop &L);
  /// @brief Recursively mark a block by_all.
  /// @param[in] BB Block to mark.
  void markByAll(llvm::BasicBlock &BB);

  /// @brief Find join points of a block.
  /// @param[in] src Starting block
  /// @return List of blocks that have a disjoint path from the starting block.
  llvm::DenseSet<llvm::BasicBlock *> joinPoints(llvm::BasicBlock &src) const;
  /// @brief Find escape points of a divergent loop.
  ///
  /// Escape points are loop exit blocks from which some work-items may leave
  /// through because of a divergent branch.
  /// @param[in] src Divergent branch
  /// @param[in] L Divergent loop
  /// @return List of exit blocks some work-item may leave through.
  llvm::DenseSet<llvm::BasicBlock *> escapePoints(const llvm::BasicBlock &src,
                                                  const llvm::Loop &L) const;

  /// @brief the Function the analysis was run on
  llvm::Function &F;
  /// @brief AM FunctionAnalysisManager providing analyses.
  llvm::FunctionAnalysisManager &AM;

  /// @brief Basic block tag mappings.
  DenseBBMap BBMap;
  /// @brief Loop tag mappings.
  DenseLoopMap LMap;

  /// @brief Storage for the Basic Block Tags
  std::vector<BasicBlockTag> basicBlockTags;
  /// @brief The number of blocks in the DCBI ordering.
  size_t numOrderedBlocks = 0;

  /// @brief List of Loop Tags ordered by loop depth
  llvm::SmallVector<LoopTag *, 16> loopOrdering;

  /// @brief Blocks that have a divergent branch.
  std::vector<llvm::BasicBlock *> divCausingBlocks;

  /// @brief Blocks with uniform conditions that must be considered div causing
  ///        because they have a join point of a div causing block as their
  ///        successor.
  llvm::DenseSet<llvm::BasicBlock *> fakeDivCausingBlocks;
};

/// @brief Analysis that determines divergent blocks, i.e. program points
///        that must not be skipped during SIMD execution.
class DivergenceAnalysis : public llvm::AnalysisInfoMixin<DivergenceAnalysis> {
  friend llvm::AnalysisInfoMixin<DivergenceAnalysis>;

 public:
  /// @brief Create a new analysis object.
  DivergenceAnalysis() = default;

  /// @brief Type of result produced by the analysis.
  using Result = DivergenceResult;

  /// @brief Determine which values in the function are uniform and which are
  /// potentially varying.
  ///
  /// @param[in] F Function to analyze.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  ///
  /// @return Analysis result for the function.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Divergence analysis"; }

 private:
  /// @brief Unique identifier for the pass.
  static llvm::AnalysisKey Key;
};
}  // namespace vecz

#endif  // VECZ_ANALYSIS_DIVERGENCE_ANALYSIS_H_INCLUDED
