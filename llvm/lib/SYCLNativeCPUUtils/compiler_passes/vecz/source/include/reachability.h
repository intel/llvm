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
/// @brief A utility class to speed of reachability queries on a CFG

#ifndef VECZ_REACHABILITY_H_INCLUDED
#define VECZ_REACHABILITY_H_INCLUDED

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <vector>

namespace llvm {
class BasicBlock;
class DominatorTree;
class Function;
class LoopInfo;
class PostDominatorTree;
}  // namespace llvm

namespace vecz {

/// @brief A data structure to handle reachability queries
class Reachability {
 public:
  /// @brief Construct the Reachability computation from a Dominator Tree
  ///        and a Post-Dominator Tree, that are used to speed up the queries.
  /// @param[in] DT the Dominator Tree
  /// @param[in] PDT the Post-Dominator Tree
  /// @param[in] LI the Loop Info
  Reachability(llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
               llvm::LoopInfo &LI);

  /// @brief Destructor
  ~Reachability() = default;

  /// @brief Computes a new data structure from the provided block tag list,
  ///        overwriting any data that was already present.
  ///
  /// Back edges are disregarded during this process.
  void recalculate(llvm::Function &F);

  /// @brief Computes a new data structure from the provided block tag list,
  ///        only if the structure is currently empty. Otherwise, does nothing.
  void update(llvm::Function &F);

  /// @brief Clears the data structure.
  ///
  /// Updating the underlying CFG invalidates the Reachability computations,
  /// so it is required to clear the data ready to accept a new CFG.
  void clear();

  /// @brief Checks the internal consistency of the computed data structure.
  bool validate() const;

  /// @brief Check if a block is reachable from another.
  ///
  /// @param[in] from the BasicBlock to start from
  /// @param[in] to the BasicBlock we are trying to reach
  ///
  /// @return True if "to" is reachable from "from"
  bool isReachable(llvm::BasicBlock *from, llvm::BasicBlock *to) const;

 private:
  /// @brief Internal implementation of isReachable
  ///
  /// @param[in] from the graph node index to start from
  /// @param[in] to the graph node index we are trying to reach
  ///
  /// @return True if "to" is reachable from "from"
  bool isReachableImpl(size_t from, size_t to) const;

  /// @brief The Dominator Tree
  llvm::DominatorTree &DT;
  /// @brief The Post-Dominator Tree
  llvm::PostDominatorTree &PDT;
  /// @brief The Loop Info, used to determine back-edges
  llvm::LoopInfo &LI;

  /// @brief Node structure containing implementational details
  ///        computed and used by the algorithm.
  struct Rnode {
    size_t X = 0;
    size_t Y = 0;
    size_t dom = 0;
    size_t postDom = 0;
    unsigned predTmp = 0;
    unsigned predecessors = 0;
    llvm::SmallVector<size_t, 2> successors;
  };

  /// @brief The list of graph nodes that encode the graph.
  std::vector<Rnode> graph;

  /// @brief A mapping between BasicBlock pointers and graph node indices.
  llvm::DenseMap<llvm::BasicBlock *, size_t> indexMap;
};
}  // namespace vecz

#endif  // VECZ_REACHABILITY_H_INCLUDED
