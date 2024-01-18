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

#include "analysis/divergence_analysis.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <multi_llvm/multi_llvm.h>

#include <algorithm>
#include <memory>

#include "analysis/uniform_value_analysis.h"
#include "debugging.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

namespace {
using RPOT = ReversePostOrderTraversal<Function *>;
}  // namespace

BlockQueue::BlockQueue(const DivergenceResult &dr,
                       const DenseSet<BasicBlock *> &blocks)
    : DR(dr) {
  indices.reserve(blocks.size());
  for (auto *const BB : blocks) {
    indices.push_back(DR.getTagIndex(BB));
  }

  // Note that make_heap builds a Max heap, so we use `std::greater` to get a
  // Min heap.
  std::make_heap(indices.begin(), indices.end(), std::greater<index_type>());
}

const BasicBlockTag &BlockQueue::pop() {
  assert(!indices.empty() && "Trying to pop from an empty BlockQueue");
  std::pop_heap(indices.begin(), indices.end(), std::greater<index_type>());
  const auto popped_index = indices.back();
  indices.pop_back();

  return DR.getBlockTag(popped_index);
}

void BlockQueue::push(size_t index) {
  indices.push_back(index);
  std::push_heap(indices.begin(), indices.end(), std::greater<index_type>());
}

void BlockQueue::push(const BasicBlock *bb) {
  indices.push_back(DR.getTagIndex(bb));
  std::push_heap(indices.begin(), indices.end(), std::greater<index_type>());
}

DivergenceResult::DivergenceResult(Function &F, FunctionAnalysisManager &AM)
    : F(F), AM(AM) {}

size_t DivergenceResult::getTagIndex(const llvm::BasicBlock *BB) const {
  assert(BB && "Trying to get the tag of a null BasicBlock");
  auto iter = BBMap.find(BB);
  assert(iter != BBMap.end() && "BasicBlock tag is not defined");
  return iter->second;
}

BasicBlockTag &DivergenceResult::getOrCreateTag(BasicBlock *BB) {
  assert(BB && "Trying to get the tag of a null BasicBlock");
  const auto &result = BBMap.try_emplace(BB, basicBlockTags.size());
  if (result.second) {
    // It's a new map entry, so create the new tag and return it.
    basicBlockTags.emplace_back();
    auto &tag = basicBlockTags.back();
    tag.BB = BB;
    return tag;
  }
  // Return the indexed tag.
  return basicBlockTags[result.first->second];
}

LoopTag &DivergenceResult::getTag(const Loop *L) const {
  assert(L && "Trying to get the tag of a null loop");
  auto iter = LMap.find(L);
  assert(iter != LMap.end() && "Loop tag is not defined");
  return *iter->second;
}

LoopTag &DivergenceResult::getOrCreateTag(Loop *L) {
  assert(L && "Trying to get or create the tag of a null loop");
  auto &tag = LMap[L];
  if (!tag) {
    tag = std::make_unique<LoopTag>();
    tag->loop = L;
  }
  return *tag;
}

bool DivergenceResult::hasFlag(const BasicBlock &BB,
                               BlockDivergenceFlag F) const {
  return (getTag(&BB).divergenceFlag & F) == F;
}

BlockDivergenceFlag DivergenceResult::getFlag(const BasicBlock &BB) const {
  return getTag(&BB).divergenceFlag;
}

void DivergenceResult::setFlag(const BasicBlock &BB, BlockDivergenceFlag F) {
  auto &tag = getTag(&BB);
  tag.divergenceFlag = static_cast<BlockDivergenceFlag>(tag.divergenceFlag | F);
}

void DivergenceResult::clearFlag(const BasicBlock &BB, BlockDivergenceFlag F) {
  auto &tag = getTag(&BB);
  tag.divergenceFlag =
      static_cast<BlockDivergenceFlag>(tag.divergenceFlag & ~F);
}

bool DivergenceResult::isDivCausing(const BasicBlock &BB) const {
  return (hasFlag(BB, BlockDivergenceFlag::eBlockHasDivergentBranch) ||
          hasFlag(BB, BlockDivergenceFlag::eBlockHasDivergentBranchFake));
}

bool DivergenceResult::isDivergent(const BasicBlock &BB) const {
  return hasFlag(BB, BlockDivergenceFlag::eBlockIsDivergent);
}

bool DivergenceResult::isOptional(const BasicBlock &BB) const {
  return !isDivergent(BB);
}

bool DivergenceResult::isByAll(const BasicBlock &BB) const {
  return hasFlag(BB, BlockDivergenceFlag::eBlockIsByAll);
}

bool DivergenceResult::isBlend(const BasicBlock &BB) const {
  return hasFlag(BB, BlockDivergenceFlag::eBlockIsBlend);
}

bool DivergenceResult::isUniform(const BasicBlock &BB) const {
  return hasFlag(BB, BlockDivergenceFlag::eBlockIsUniform);
}

bool DivergenceResult::hasFlag(const Loop &L, LoopDivergenceFlag F) const {
  return (getTag(&L).divergenceFlag & F) == F;
}

LoopDivergenceFlag DivergenceResult::getFlag(const Loop &L) const {
  return getTag(&L).divergenceFlag;
}

void DivergenceResult::setFlag(const Loop &L, LoopDivergenceFlag F) {
  auto &tag = getTag(&L);
  tag.divergenceFlag = static_cast<LoopDivergenceFlag>(tag.divergenceFlag | F);
}

void DivergenceResult::clearFlag(const Loop &L, LoopDivergenceFlag F) {
  auto &tag = getTag(&L);
  tag.divergenceFlag = static_cast<LoopDivergenceFlag>(tag.divergenceFlag & ~F);
}

bool DivergenceResult::computeBlockOrdering(DominatorTree &DT) {
  LLVM_DEBUG(dbgs() << "Divergence Analysis: COMPUTE BLOCK ORDERING\n");

  // The DCBI (Dominance Compact Block Indexing) is a topological ordering of
  // the basic blocks that is also dominance compact, that is, an ordering such
  // that for any block A, every block that A dominates follows in a contiguous
  // subsequence in the ordering. To construct this, we gather a reverse post-
  // order traversal over the CFG, and then a depth-first traversal over the
  // dominator tree, ordering each node's children according to the previously
  // calculated reverse post-order. We need to take special care of loop exits,
  // however, since where a loop exits from some block other than a latch,
  // the dominator tree traversal can erroneously order it inside of the loop.
  // To prevent this, we store up exit blocks until we have processed all
  // the blocks at the current loop level.

  struct DCnode {
    BasicBlock *BB;
    unsigned depth = 0;
  };
  std::vector<DCnode> graph;
  llvm::DenseMap<llvm::BasicBlock *, unsigned> indexMap;

  indexMap.reserve(F.size());
  {
    // Note that a post-order traversal of the CFG does not include any blocks
    // with no predecessors, other than the entry block.
    unsigned index = 0;
    for (auto *const BB : RPOT(&F)) {
      indexMap[BB] = index++;
      graph.emplace_back();
      graph.back().BB = BB;

      if (const auto *const LTag = getTag(BB).loop) {
        graph.back().depth = LTag->loop->getLoopDepth();
      }
    }
  }

  // Do a depth-first traversal of the dominator tree
  SmallVector<unsigned, 16> stack;
  stack.push_back(0);
  uint32_t pos = 0;
  const SmallVector<unsigned, 16> children;
  SmallVector<unsigned, 16> loopExits;
  while (!stack.empty()) {
    const auto u = stack.pop_back_val();
    const auto &uNode = graph[u];

    getTag(uNode.BB).pos = pos++;

    // Children in the same loop or subloops get added back to the stack.
    // Children outside of the current loop get stored up until we processed
    // everything in this loop. Note that we can accumulate exit blocks
    // from multiple points within the loop, and across multiple depth levels.
    auto *const DTNode = DT.getNode(uNode.BB);
    unsigned stacked = 0;
    for (auto *const childNode : make_range(DTNode->begin(), DTNode->end())) {
      const auto child = indexMap[childNode->getBlock()];
      auto &cNode = graph[child];
      if (cNode.depth >= uNode.depth) {
        stack.push_back(child);
        ++stacked;
      } else {
        // Note that we can exit across more than one loop level, so we need to
        // find the right place to insert it.
        auto insert = loopExits.end();
        while (insert != loopExits.begin()) {
          auto scan = insert - 1;
          if (cNode.depth < graph[*scan].depth) {
            insert = scan;
          } else {
            break;
          }
        }
        loopExits.insert(insert, child);
      }
    }
    // Sort any children added to the stack into post-order
    std::sort(stack.end() - stacked, stack.end(), std::greater<unsigned>());

    if (!loopExits.empty()) {
      const unsigned curDepth = stack.empty() ? 0 : graph[stack.back()].depth;
      const unsigned depth = std::max(curDepth, graph[loopExits.back()].depth);
      unsigned count = 0;
      while (!loopExits.empty() && depth == graph[loopExits.back()].depth) {
        stack.push_back(loopExits.pop_back_val());
        ++count;
      }

      // Sort the loop exits into post-order
      std::sort(stack.end() - count, stack.end(), std::greater<unsigned>());
    }
  }
  assert(pos == graph.size() && "Incomplete DCBI");

  reorderTags(pos);
  return true;
}

void DivergenceResult::reorderTags(size_t n) {
  numOrderedBlocks = n;

  // This is a Cycle Sort. It re-orders the tags in the tag vector according to
  // their calculated block index. Despite the two nested loops, it is O(n).
  // Out-of-range indices (pos >= n) will be left where they are, but a later
  // ordered tag might move it afterwards.
  for (size_t i = 0, n = basicBlockTags.size(); i != n; ++i) {
    auto &tag = basicBlockTags[i];
    while (tag.pos < n && tag.pos != i) {
      std::swap(tag, basicBlockTags[tag.pos]);
    }
  }

  // Rebuild the index map after sorting. Note that we can't absorb this into
  // the above loop, since an unordered tag might not be in its final position
  // until all of the ordered tags are in their correct places.
  for (size_t i = 0, n = basicBlockTags.size(); i != n; ++i) {
    BBMap[basicBlockTags[i].BB] = i;
  }
}

bool DivergenceResult::computeLoopOrdering() {
  loopOrdering.clear();
  for (const auto &pair : LMap) {
    loopOrdering.push_back(pair.second.get());
  }

  std::sort(loopOrdering.begin(), loopOrdering.end(),
            [](const LoopTag *LHS, const LoopTag *RHS) -> bool {
              return LHS->loop->getLoopDepth() < RHS->loop->getLoopDepth();
            });

  return true;
}

void DivergenceResult::markDivCausing(BasicBlock &BB, DivergenceInfo &DI,
                                      PostDominatorTree &PDT) {
  if (isDivCausing(BB)) {
    return;
  }

  divCausingBlocks.push_back(&BB);
  setFlag(BB, BlockDivergenceFlag::eBlockHasDivergentBranch);
  LLVM_DEBUG(dbgs() << "Block " << BB.getName() << " is div_causing\n");

  for (BasicBlock *succ : successors(&BB)) {
    markDivergent(*succ);
  }

  // If a block is a joint point (blend) of `BB`, then it is divergent (unless
  // it is the post-dominator of `BB`).
  const auto &joins = joinPoints(BB);
  for (BasicBlock *const join : joins) {
    setFlag(*join, BlockDivergenceFlag::eBlockIsBlend);
    LLVM_DEBUG(dbgs() << "\tBlock " << join->getName() << " is blend\n");

    if (!PDT.dominates(join, &BB)) {
      markDivergent(*join);
    }

    for (BasicBlock *const pred : predecessors(join)) {
      // If at least 2 successors of `pred` are join points of `BB`, then mark
      // `pred` as a fake div causing block because its successors may be
      // executed by multiple work-items.
      if (std::count_if(
              succ_begin(pred), succ_end(pred),
              [&joins](BasicBlock *succ) { return joins.count(succ); }) > 1) {
        fakeDivCausingBlocks.insert(pred);
      }
    }

    // Join points of divergent branches need their PHIs marked varying.
    DI.insert(join);
  }
}

void DivergenceResult::markDivLoopDivBlocks(BasicBlock &BB, Loop &L,
                                            DivergenceInfo &DI) {
  markDivergent(L);

  // Find loop exits through which some work-items may leave the loop while
  // others keep iterating over it. These exit blocks can be reached from the
  // div_causing block before reaching the latch because the divergent path
  // cannot fully reconverge before leaving the loop (since the loop is
  // divergent).
  SmallVector<BasicBlock *, 1> exits;
  L.getExitBlocks(exits);
  const auto &divergentExits = escapePoints(BB, L);
  for (BasicBlock *E : exits) {
    if (divergentExits.count(E)) {
      markDivergent(*E);
    }
    // All loop exits of a divergent loop need their PHIs marked varying.
    DI.insert(E);
  }

  // The latch of a divergent loop is divergent.
  markDivergent(*L.getLoopLatch());
}

void DivergenceResult::markDivergent(const BasicBlock &BB) {
  if (!isDivergent(BB)) {
    setFlag(BB, BlockDivergenceFlag::eBlockIsDivergent);
    LLVM_DEBUG(dbgs() << "\tBlock " << BB.getName() << " is divergent\n");
  }
}

void DivergenceResult::markDivergent(const Loop &L) {
  if (!getTag(&L).isLoopDivergent()) {
    setFlag(L, LoopDivergenceFlag::eLoopIsDivergent);
    LLVM_DEBUG(dbgs() << "\tLoop " << L.getName() << " is divergent\n");
  }
}

void DivergenceResult::markByAll(BasicBlock &src) {
  Function &F = *src.getParent();
  const DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  const PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  BlockQueue queue(*this);
  queue.push(&src);

  while (!queue.empty()) {
    auto &BBTag = queue.pop();
    auto *const BB = BBTag.BB;

    if (isByAll(*BB)) {
      continue;
    }

    const bool isHeaderDivLoop =
        BBTag.isLoopHeader() && BBTag.loop->isLoopDivergent();
    // If BB is a loop header, it can only be marked by_all if its loop does not
    // diverge.
    if (!isHeaderDivLoop) {
      setFlag(*BB, BlockDivergenceFlag::eBlockIsByAll);
      LLVM_DEBUG(dbgs() << "Block " << BB->getName() << " is by_all\n");
    }

    SmallVector<BasicBlock *, 2> descendants;
    DT.getDescendants(BB, descendants);

    // For all descendants `D` of `BB` that post-dominate `BB`, `D` is by_all.
    for (BasicBlock *D : descendants) {
      if (D != BB) {
        if (PDT.dominates(D, BB)) {
          const auto DIndex = getTagIndex(D);
          const auto *const DLoopTag = basicBlockTags[DIndex].loop;
          // If we are not in a loop, or the loop we live in does not diverge
          // nor does the one englobing us if it exists, then mark by_all.
          Loop *parentLoop;
          if (!DLoopTag || (!DLoopTag->isLoopDivergent() &&
                            (!(parentLoop = DLoopTag->loop->getParentLoop()) ||
                             isByAll(*parentLoop->getHeader())))) {
            queue.push(DIndex);
          }
        }
      }
    }

    // For all descendants `D` of `BB` that do not post-dominate `BB`, `D` is
    // by_all if all predecessors of `D` are by_all.
    //
    // If BB is a divergent branch, it cannot propagate by_all to its
    // successors.
    if (!isHeaderDivLoop && !isDivCausing(*BB)) {
      for (BasicBlock *D : descendants) {
        if (D != BB) {
          if (!PDT.dominates(D, BB)) {
            if (std::all_of(
                    pred_begin(D), pred_end(D),
                    [this](BasicBlock *pred) { return isByAll(*pred); })) {
              queue.push(D);
            }
          }
        }
      }
    }
  }
}

bool DivergenceResult::isReachable(BasicBlock *src, BasicBlock *dst,
                                   bool allowLatch) const {
  DenseSet<BasicBlock *> visited;
  std::vector<BasicBlock *> worklist;

  worklist.push_back(src);
  visited.insert(src);

  while (!worklist.empty()) {
    BasicBlock *BB = worklist.back();
    worklist.pop_back();

    if (BB == dst) {
      return true;
    }

    const auto &BBTag = getTag(BB);
    for (BasicBlock *succ : successors(BB)) {
      if (!allowLatch && BBTag.isLoopBackEdge(succ)) {
        continue;
      }
      if (visited.insert(succ).second) {
        worklist.push_back(succ);
      }
    }
  }

  return false;
}

DenseSet<BasicBlock *> DivergenceResult::joinPoints(BasicBlock &src) const {
  if (src.getTerminator()->getNumSuccessors() < 2) {
    return {};
  }

  Function &F = *src.getParent();
  const PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  DenseMap<const BasicBlock *, const BasicBlock *> defMap;
  DenseSet<BasicBlock *> joins;

  BlockQueue queue(*this);

  auto schedule = [&defMap, &joins, &queue](BasicBlock *block,
                                            const BasicBlock *defBlock) {
    auto defIt = defMap.find(block);
    // First time we meet this block; not a join (yet).
    if (defIt == defMap.end()) {
      queue.push(block);
      defMap.insert({block, defBlock});
    } else if (defIt->second != defBlock) {
      // We've found a block that has two different incoming definitions; it is
      // a join point.
      joins.insert(block);
    }
  };

  for (BasicBlock *const succ : successors(&src)) {
    schedule(succ, succ);
  }

  auto *Node = PDT.getNode(&src);
  assert(Node && "Could not get node");
  auto *IDom = Node->getIDom();
  assert(IDom && "Could not get IDom");
  BasicBlock *PIDom = IDom->getBlock();
  assert(PIDom && "Could not get block");

  while (!queue.empty()) {
    auto &curTag = queue.pop();
    BasicBlock *cur = curTag.BB;

    if (cur == PIDom) {
      continue;
    }

    const BasicBlock *const defBlock = defMap.find(cur)->second;

    const auto *const curLTag = curTag.loop;
    // If the successor is the header of a nested loop pretend its a single
    // node with the loop's exits as successors.
    if (curLTag && curLTag->header == cur) {
      SmallVector<BasicBlock *, 2> exits;
      curLTag->loop->getUniqueExitBlocks(exits);
      for (BasicBlock *const exit : exits) {
        if (exit == &src) {
          continue;
        }
        schedule(exit, defBlock);
      }
    } else {
      // the successors are either on the same loop level or loop exits
      for (BasicBlock *const succ : successors(cur)) {
        if (succ == &src) {
          continue;
        }
        schedule(succ, defBlock);
      }
    }
  }

  return joins;
}

DenseSet<BasicBlock *> DivergenceResult::escapePoints(const BasicBlock &src,
                                                      const Loop &L) const {
  const LoopTag &LTag = getTag(&L);

  DenseSet<BasicBlock *> divergentExits;

  DenseSet<const BasicBlock *> visited;
  BlockQueue queue(*this);

  queue.push(&src);
  visited.insert(&src);

  while (!queue.empty()) {
    const auto &BBTag = queue.pop();
    auto *const BB = BBTag.BB;

    // We found a divergent loop exit.
    if (!L.contains(BB)) {
      divergentExits.insert(BB);
      continue;
    }

    bool allowLatch = true;
    auto *const loopTag = BBTag.loop;
    // 'BB' is a backedge
    if (loopTag && loopTag->latch == BB) {
      if (loopTag == &LTag) {
        // `BB` is the latch of the current loop; forbid the backedge.
        allowLatch = false;
      } else {
        // Otherwise, forbid the backedge only if none of the remaining blocks
        // in the queue belong to `L`, in which case no exit block starting
        // from the header of the nested loop can be divergent.
        allowLatch =
            std::any_of(queue.begin(), queue.end(), [this, &L](size_t index) {
              return L.contains(basicBlockTags[index].BB);
            });
      }
    }

    for (BasicBlock *succ : successors(BB)) {
      if (BBTag.isLoopBackEdge(succ) && !allowLatch) {
        continue;
      }
      if (visited.insert(succ).second) {
        queue.push(succ);
      }
    }
  }

  return divergentExits;
}

////////////////////////////////////////////////////////////////////////////////

llvm::AnalysisKey DivergenceAnalysis::Key;

DivergenceResult DivergenceAnalysis::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &AM) {
  DivergenceResult Res(F, AM);

  LLVM_DEBUG(dbgs() << "DIVERGENCE ANALYSIS\n");
  Res.basicBlockTags.reserve(F.size() * 4);

  // Prepare the BasicBlockTags.
  const LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  for (BasicBlock &BB : F) {
    // Create BB info entries.
    BasicBlockTag &BBTag = Res.getOrCreateTag(&BB);

    // Update loop info.
    if (Loop *L = LI.getLoopFor(&BB)) {
      if (!BBTag.loop) {
        BBTag.loop = &Res.getOrCreateTag(L);
        BBTag.loop->latch = L->getLoopLatch();
        BBTag.loop->header = L->getHeader();
        BBTag.loop->preheader = L->getLoopPreheader();
      }
    }
  }

  // Find loop live values and update loop exit information.
  Res.computeLoopOrdering();
  for (auto *const LTag : Res.loopOrdering) {
    SmallVector<BasicBlock *, 1> loopExitBlocks;
    LTag->loop->getExitBlocks(loopExitBlocks);
    for (BasicBlock *BB : loopExitBlocks) {
      auto &BBTag = Res.getTag(BB);
      // If BB already leaves a loop, update it if the previous loop is nested
      // in the current.
      if (BBTag.outermostExitedLoop) {
        if (BBTag.outermostExitedLoop->loop->getLoopDepth() >
            LTag->loop->getLoopDepth()) {
          BBTag.outermostExitedLoop = LTag;
        }
      } else {
        BBTag.outermostExitedLoop = LTag;
      }

      // LoopSimplify pass has already converted SSA form to LCSSA from.
      // Let's use lcssa phi nodes to find loop live variables like llvm loop
      // vectorizer.
      // LoopSimplify pass is added on PreparationPass of vectorizer.cpp.
      //
      // See head comment on lib/Transforms/Utils/LCSSA.cpp
      for (Instruction &I : *BB) {
        if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
          // lcssa phi has incoming values defined in the loop.
          for (Value *incoming : PHI->incoming_values()) {
            if (Instruction *incomingInst = dyn_cast<Instruction>(incoming)) {
              if (LTag->loop->contains(incomingInst->getParent())) {
                LTag->loopLiveValues.insert(incoming);
                LLVM_DEBUG(dbgs() << *incoming << " is a loop live value of "
                                  << LTag->loop->getName() << "\n");
              }
            }
          }
        }
      }
    }
  }

  // From the UVA, we know which conditions are varying which allows us to
  // find divergent branches.
  // Moreover, from divergent branches - and therefore from divergent paths -
  // we can find more varying values that are computed on those divergent paths.
  // The latter allows us to find more divergent branches, and so on...
  // We take a local copy of the UVR because it is not good to modify one
  // analysis result from another analysis. However, after Control Flow
  // Conversion has been run, all control flow divergence is converted into
  // non-uniform dataflow so any subsequent run of the UVA is still correct.
  auto UVR = AM.getResult<UniformValueAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  Res.computeBlockOrdering(DT);

  std::vector<std::pair<BasicBlock *, Value *>> uniformBranches;
  uniformBranches.reserve(F.size() - 1u);
  for (BasicBlock &BB : F) {
    if (BranchInst *B = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (B->isConditional()) {
        uniformBranches.push_back({&BB, B->getCondition()});
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
      uniformBranches.push_back({&BB, SI->getCondition()});
    }
  }

  while (!uniformBranches.empty()) {
    // Partition the list so all the varying branches are grouped at the end.
    const auto varyingBranches =
        std::partition(uniformBranches.begin(), uniformBranches.end(),
                       [&UVR](std::pair<BasicBlock *, Value *> &p) -> bool {
                         return !UVR.isVarying(p.second);
                       });

    // Process all the varying branches.
    DivergenceInfo divergenceInfo;
    for (auto it = varyingBranches; it != uniformBranches.end(); ++it) {
      BasicBlock *BB = it->first;

      // Find blocks diverged by varying branch block.
      Res.markDivCausing(*BB, divergenceInfo, PDT);

      if (const auto *const LTag = Res.getTag(BB).loop) {
        Loop *L = LTag->loop;
        while (L) {
          // If BB is a varying branch, mark the loop as diverging if any two
          // instances of a SIMD group can leave the loop over different exit
          // edges and/or in different iterations. This means that BB cannot
          // be postdominated by any block of L.
          auto *Node = PDT.getNode(BB);
          assert(Node && "Could not get node");
          auto *IDom = Node->getIDom();
          assert(IDom && "Could not get IDom");
          BasicBlock *PIDom = IDom->getBlock();
          if (!L->contains(PIDom)) {
            Res.markDivLoopDivBlocks(*BB, *L, divergenceInfo);
          } else {
            // If the loop does not diverge because of `BB`, none of its
            // parent loops can diverge either.
            break;
          }
          L = L->getParentLoop();
        }
      }
    }

    // Remove all the varying branches from the end of the list.
    uniformBranches.erase(varyingBranches, uniformBranches.end());

    // PHIs defined in join point of divergent branches and in exit blocks of
    // divergent loops are varying.
    bool updated = false;
    for (BasicBlock *BB : divergenceInfo) {
      const bool exitedLoop = Res.getTag(BB).outermostExitedLoop;
      for (Instruction &I : *BB) {
        if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
          // Loop exits might have constant phi nodes (lcssa value).
          if (exitedLoop || !PHI->hasConstantOrUndefValue()) {
            if (!UVR.isVarying(&I)) {
              updated = true;
              UVR.markVaryingValues(&I);
              LLVM_DEBUG(dbgs()
                         << I.getName() << " is a varying instruction\n");
            }
          }
        } else {
          break;
        }
      }
    }
    if (!updated) {
      // We made no updates, so we processed all the varying branches.
      break;
    }
  }

  // All blocks that are predecessors of join points of div causing blocks and
  // have a uniform condition must be marked as fake div causing blocks because
  // divergence may have occurred at the div causing block and we must make sure
  // we execute all paths that lead to the join point.
  for (BasicBlock *BB : Res.fakeDivCausingBlocks) {
    if (BB->getTerminator()->getNumSuccessors() > 1 && !Res.isDivCausing(*BB)) {
      Res.setFlag(*BB, BlockDivergenceFlag::eBlockHasDivergentBranchFake);
      LLVM_DEBUG(dbgs() << "Found fake div causing block " << BB->getName()
                        << "\n");
      // Because we have marked `BB` as a target for linearization, its join
      // points must be marked as `blend` because they may lose some
      // predecessors during the rewiring.
      for (BasicBlock *join : Res.joinPoints(*BB)) {
        Res.setFlag(*join, BlockDivergenceFlag::eBlockIsBlend);
        LLVM_DEBUG(dbgs() << "\tBlock " << join->getName() << " is blend\n");
      }
    }
  }

  // By definition, the entry block is by_all.
  Res.markByAll(F.getEntryBlock());

  return Res;
}
