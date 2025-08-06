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
/// @brief BOSCC control flow transformation.
///
/// Style guideline 004 exemption note: This inner class declaration is in its
/// own header file, because it's quite large.

#ifndef VECZ_CONTROL_FLOW_BOSCC_H_INCLUDED
#define VECZ_CONTROL_FLOW_BOSCC_H_INCLUDED

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <utility>
#include <vector>

#include "transform/control_flow_conversion_pass.h"

namespace llvm {
class Instruction;
class BasicBlock;
class Function;
class Loop;
}  // namespace llvm

namespace vecz {

class LivenessResult;

class ControlFlowConversionState::BOSCCGadget final {
 public:
  BOSCCGadget(ControlFlowConversionState &Pass)
      : PassState(Pass),
        F(Pass.F),
        AM(Pass.AM),
        DT(Pass.DT),
        PDT(Pass.PDT),
        LI(Pass.LI),
        DR(Pass.DR),
        RC(Pass.RC.get()) {}

  /// @brief Region of code that will remain uniform after vectorization.
  ///
  /// Such regions won't have heir instructions predicated. A UniformRegion
  /// is delimited by a single-entry-single-exit region and is represented
  /// by the blocks it contains.
  struct UniformRegion final {
    /// @brief Predicated blocks duplicated in the region.
    llvm::DenseSet<llvm::BasicBlock *> predicatedBlocks;
    /// @brief Uniform blocks created in the region.
    llvm::DenseSet<llvm::BasicBlock *> uniformBlocks;
    /// @brief Divergent branches that need a connection from the uniform
    /// region.
    std::vector<llvm::BasicBlock *> divergentBranches;
    /// @brief The entry block of the uniform region.
    llvm::BasicBlock *entryBlock;
    /// @brief The exit block of the uniform region.
    llvm::BasicBlock *exitBlock;

    /// @brief Mapping between a connection point of a predicated region
    ///        and the blend points of that region impacted by the former.
    ///
    /// Said "impacted blocks" are blocks with more than one predecessors that
    /// need to have blend instructions because instructions defined within
    /// that region may no longer dominate said "impacted blocks".
    llvm::DenseMap<llvm::BasicBlock *, llvm::SmallVector<llvm::BasicBlock *, 2>>
        blendPoints;

    /// @brief It stores up information about the connection points while
    ///        the CFG is being updated, to be applied afterwards.
    struct ConnectionInfo {
      llvm::BasicBlock *connectionPoint;
      std::pair<llvm::BasicBlock *, llvm::BasicBlock *> incoming;
    };

    /// @brief The list of ConnectionInfos to be applied at finalization.
    std::vector<ConnectionInfo> connections;

    /// @brief It stores up information about new blocks created to contain
    ///        blend LCSSA PHI nodes, so they can be created after the CFG
    ///        has been updated.
    struct StoreBlock {
      llvm::BasicBlock *connectionPoint;
      llvm::BasicBlock *target;
      llvm::BasicBlock *runtimeCheckerBlock;
    };

    /// @brief The list of blend `StoreBlocks` to be applied at finalization.
    llvm::SmallVector<StoreBlock, 4> storeBlocks;

    /// @brief Find if a predicated block belongs to this region.
    /// @param[in] B Block to look for in the region
    /// @return Whether the block belong to the region or not.
    bool contains(llvm::BasicBlock *B) const {
      return predicatedBlocks.count(B);
    }
  };
  /// @brief List of all duplicated uniform regions.
  using UniformRegions = std::vector<UniformRegion>;

  /// @brief Create uniform regions to duplicate the blocks within such
  /// regions.
  ///
  /// This allows to retain their uniform version to skip divergent branches
  /// when the entry mask of a div causing block is dynamically uniform (i.e.
  /// all true or all false). Nested uniform regions need not be duplicated
  /// multiple times.
  ///
  /// @return true if no problem occurred, false otherwise.
  bool duplicateUniformRegions();

  /// @brief Connect the BOSCC regions.
  /// @return true if no problem occured, false otherwise.
  bool connectBOSCCRegions();

  /// @brief Get the uniform version of 'B'.
  /// @param[in] B The predicated block whose uniform version we want.
  /// @return A uniform block if it exists, nullptr otherwise.
  llvm::BasicBlock *getBlock(llvm::BasicBlock *B);
  /// @brief Get the uniform version of 'L'.
  /// @param[in] L The predicated loop whose uniform version we want.
  /// @return A uniform loop if it exists, nullptr otherwise.
  llvm::Loop *getLoop(llvm::Loop *L);

  /// @brief Get the region entry blocks that have not been duplicated.
  /// @param[out] blocks SmallVector to hold the result
  void getUnduplicatedEntryBlocks(
      llvm::SmallVectorImpl<llvm::BasicBlock *> &blocks) const;

  /// @brief Create an entry in the VMap so that 'uni' becomes a uniform
  ///        equivalent of 'pred'.
  /// @param[in] pred Predicate value
  /// @param[in] uni Uniform value
  /// @param[in] needsMapping Whether 'uni' needs to me remapped
  void createReference(llvm::Value *pred, llvm::Value *uni,
                       bool needsMapping = false);
  /// @brief Add an entry in the VMap so that the uniform equivalent of
  ///        'old' becomes the uniform equivalent of 'pred' as well.
  /// @param[in] pred Predicate value
  /// @param[in] old Predicate value whose uniform equivalent we want
  void addReference(llvm::Value *pred, llvm::Value *old);
  /// @brief Add a new block to all the regions the reference block is part
  /// of.
  /// @param[in] newB New block
  /// @param[in] refB Rference block
  void addInRegions(llvm::BasicBlock *newB, llvm::BasicBlock *refB);

  /// @brief Link the masks of the predicated regions to the uniform regions.
  /// @return true on success, false on failure.
  bool linkMasks();

  /// @brief Retrieve the uniform version, if one exists, of predicatedV
  ///        defined in src.
  /// @param[in] predicatedV The predicated value whose uniform version we
  ///            want to get.
  /// @return the uniform version if it exists, null otherwise.
  llvm::Value *getUniformV(llvm::Value *predicatedV);
  /// @brief Update the value a uniform value should be a duplicate of.
  /// @param[in] from The old value
  /// @param[in] to The new value
  void updateValue(llvm::Value *from, llvm::Value *to);

  /// @brief Clean up redundant PHI nodes created by BOSCC.
  /// @return true if no problem occured, false otherwise.
  bool cleanUp();

 private:
  ControlFlowConversionState &PassState;
  llvm::Function &F;
  llvm::FunctionAnalysisManager &AM;
  llvm::DominatorTree *DT = nullptr;
  llvm::PostDominatorTree *PDT = nullptr;
  llvm::LoopInfo *LI = nullptr;
  DivergenceResult *DR = nullptr;
  Reachability *RC = nullptr;

  /// @brief Mapping between the uniform version and the predicated version
  ///        of the BOSCC. This is useful to keep information between both
  ///        versions shared, such as exit masks.
  llvm::ValueToValueMapTy VMap;

  /// @brief Mapping between the predicated version and the uniform version
  ///        of the BOSCC loops.
  llvm::DenseMap<const llvm::Loop *, llvm::Loop *> LMap;

  UniformRegions uniformRegions;

  /// @brief Original edges of the CFG. Used to connect the uniform regions
  ///        to their predicated version.
  llvm::DenseMap<llvm::BasicBlock *, llvm::SmallVector<llvm::BasicBlock *, 2>>
      uniformEdges;

  /// @brief Mapping between a block from which a value should be replaced by
  ///        its blended value.
  using URVBlender =
      std::vector<std::pair<llvm::BasicBlock *,
                            std::pair<llvm::Value *, llvm::Instruction *>>>;

  URVBlender URVB;

  LivenessResult *liveness = nullptr;

  /// @brief Create uniform regions
  /// @return true if no problem occurred, false otherwise.
  bool createUniformRegions(
      const llvm::DenseSet<llvm::BasicBlock *> &noDuplicateBlocks);
  /// @brief Duplicate a loop, creating a new looptag and updating all the
  ///        relevant information.
  /// @param[in] L The loop to duplicate
  /// @return true if no problem occurred, false otherwise.
  bool duplicateUniformLoops(llvm::Loop *L);

  /// @brief Connect the uniform blocks that belong to the uniform region
  /// @param[in] region Uniform region we are connecting
  /// @param[in] predicatedB Div causing block in the predicated version
  /// @param[in] uniformB Div causing block in the uniform version
  /// @return true if no problem occured, false otherwise.
  bool connectUniformRegion(UniformRegion &region,
                            llvm::BasicBlock *predicatedB,
                            llvm::BasicBlock *uniformB);

  /// @brief Blend uniform region instructions into the predicated region
  ///        connection point 'CP'.
  /// @param[in] CP Connection point between a uniform and predicated region.
  /// @param[in] incoming Predicated and uniform incoming block of 'CP'.
  /// @return true if no problem occured, false otherwise.
  bool blendConnectionPoint(
      llvm::BasicBlock *CP,
      const std::pair<llvm::BasicBlock *, llvm::BasicBlock *> &incoming);

  /// @brief Apply all the changes stored up by `connectUniformRegion`
  ///        and `blendConnectionPoint` once the CFG has been fully updated.
  /// @return true if no problem occured, false otherwise.
  bool blendFinalize();

  /// @brief Update blend values in loop headers.
  /// @param[in] LTag Loop whose blend values we update
  /// @param[in] from The value we want to update
  /// @param[in] to The value we update 'from' with.
  /// @return true if no problem occured, false otherwise.
  bool updateLoopBlendValues(LoopTag *LTag, llvm::Instruction *from,
                             llvm::Instruction *to);

  /// @brief Generate a block ordering.
  ///
  /// This ordering differs a little bit from the one in
  /// ControlFlowConversionPass as we must process all the blocks that belong
  /// in the same uniform region at once.
  ///
  /// @returns true if no errors occurred.
  bool computeBlockOrdering();
};
}  // namespace vecz

#endif  // VECZ_CONTROL_FLOW_BOSCC_H_INCLUDED
