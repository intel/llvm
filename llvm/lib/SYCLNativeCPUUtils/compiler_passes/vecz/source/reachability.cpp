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
#include "reachability.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>

#include "debugging.h"

#define DEBUG_TYPE "vecz-reachability"

// HOW IT WORKS
//
// It builds two complementary topological sorts of the supplied basic blocks,
// which it then uses to filter out obviously unreachable blocks as early as
// possible. Where we have two blocks A and B and B has any topology index
// less than that of A, then B is definitely not reachable from A. However,
// if B has a higher index, it might be (but we have to check to be sure).
//
// For details on the above approach, see "Reachability Queries in Very Large
// Graphs: A Fast Refined Online Search Approach" by
// Renê R. Veloso, Loïc Cerf, Wagner Meira Jr, Mohammed J. Zaki.
//
// It also uses data from the Dominator Tree and Post Dominator Tree, in order
// to skip ahead. If we want to know if B is reachable from A and we know
// that C dominates B, if A->C is not ruled out by the topology indices then we
// know there can be no path from A to B that does NOT go through C, therefore
// we only need to check if C is reachable from A. The same follows in reverse
// for Post Dominators.

using namespace llvm;

namespace vecz {

Reachability::Reachability(DominatorTree &p_DT, PostDominatorTree &p_PDT,
                           LoopInfo &p_LI)
    : DT(p_DT), PDT(p_PDT), LI(p_LI) {}

void Reachability::update(Function &F) {
  if (graph.empty()) {
    recalculate(F);
  }
}

void Reachability::clear() {
  indexMap.clear();
  graph.clear();
}

void Reachability::recalculate(Function &F) {
  clear();

  indexMap.reserve(F.size());
  graph.resize(F.size());
  {
    size_t i = 0;
    for (auto &BB : F) {
      indexMap[&BB] = i++;
    }
  }

  for (auto &BB : F) {
    auto &node = graph[indexMap[&BB]];

    auto *const loop = LI.getLoopFor(&BB);
    auto *const header = loop ? loop->getHeader() : nullptr;
    for (BasicBlock *succ : successors(&BB)) {
      if (succ == header) {
        continue;
      }

      const size_t succIndex = indexMap[succ];

      node.successors.push_back(succIndex);
      auto &succNode = graph[succIndex];
      ++succNode.predecessors;
    }
    std::sort(node.successors.begin(), node.successors.end());

    if (auto *DTNode = DT.getNode(&BB)) {
      if (auto *IDom = DTNode->getIDom()) {
        const size_t dom = indexMap[IDom->getBlock()];
        node.dom = dom;
      }
    }
    if (auto *PDTNode = PDT.getNode(&BB)) {
      if (auto *IPDom = PDTNode->getIDom()) {
        const size_t postDom = indexMap[IPDom->getBlock()];
        node.postDom = postDom;
      }
    }
  }

  std::vector<size_t> roots;
  size_t Xindex = 0;
  size_t Yindex = 0;

  // It would be surprising in fact if there was more than one root, because
  // we only expect a single entry block for a function, however we deal with
  // it for completeness, and in case this is required to be valid for some
  // intermediate state.
  {
    size_t i = 0;
    for (auto &node : graph) {
      if (node.successors.empty()) {
        node.postDom = ~size_t(0);
      }
      node.predTmp = node.predecessors;
      if (node.predecessors == 0) {
        roots.push_back(i);
      }
      ++i;
    }
  }
  // A copy of the roots vector so we don't need to build it again when we come
  // to construct the Y index.
  std::vector<size_t> rootsY = roots;

  while (!roots.empty()) {
    const size_t u = roots.back();
    roots.pop_back();

    auto &uNode = graph[u];
    uNode.X = Xindex++;
    for (const size_t v : uNode.successors) {
      auto &vNode = graph[v];
      if (--vNode.predTmp == 0) {
        roots.push_back(v);
      }
    }
  }

  for (auto &node : graph) {
    node.predTmp = node.predecessors;
  }
  roots.swap(rootsY);

  // Y heap represents right-most vertices (max X)
  auto cmpY = [this](size_t lhs, size_t rhs) -> bool {
    return graph[lhs].X < graph[rhs].X;
  };

  // The vector of roots has strictly decreasing X index, so it already has
  // the property of a max heap. No need to make_heap!
  while (!roots.empty()) {
    std::pop_heap(roots.begin(), roots.end(), cmpY);
    const size_t u = roots.back();
    roots.pop_back();

    auto &uNode = graph[u];
    uNode.Y = Yindex++;
    for (auto vi = uNode.successors.rbegin(), ve = uNode.successors.rend();
         vi != ve; ++vi) {
      const size_t v = *vi;
      auto &vNode = graph[v];
      if (--vNode.predTmp == 0) {
        roots.push_back(v);
        std::push_heap(roots.begin(), roots.end(), cmpY);
      }
    }
  }

  LLVM_DEBUG({
    size_t i = 0;
    for (auto &BB : F) {
      auto &node = graph[i];
      dbgs() << BB.getName() << ":\n";
      dbgs() << "[ " << node.X << ", " << node.Y << " ] : ";
      dbgs() << "( " << node.dom << ", " << node.postDom << " ) : ";
      for (const size_t s : node.successors) {
        if (graph[s].X <= graph[i].X) {
          dbgs() << "!x!";
        }
        if (graph[s].Y <= graph[i].Y) {
          dbgs() << "!y!";
        }
        dbgs() << s << "; ";
      }
      dbgs() << "\n\n";
      ++i;
    }
  });

  assert(validate() && "Topological indices not valid for reachability graph");
}

bool Reachability::validate() const {
  for (auto &node : graph) {
    for (const size_t s : node.successors) {
      if (graph[s].X <= node.X || graph[s].Y <= node.Y) {
        return false;
      }
    }
  }
  return true;
}

bool Reachability::isReachableImpl(size_t from, size_t to) const {
  DenseSet<size_t> visited;
  std::vector<size_t> worklist;

  while (true) {
    auto &nodeFrom = graph[from];
    auto &nodeTo = graph[to];

    if (nodeFrom.X > nodeTo.X || nodeFrom.Y > nodeTo.Y) {
      return false;
    }

    const size_t dom = nodeTo.dom;
    const size_t postDom = nodeFrom.postDom;
    if (dom == from || postDom == to) {
      return true;
    }

    auto &nodeDom = graph[dom];
    if (nodeFrom.X < nodeDom.X && nodeFrom.Y < nodeDom.Y) {
      to = dom;
      continue;
    }

    if (postDom != ~size_t(0)) {
      auto &nodePDom = graph[postDom];
      if (nodePDom.X < nodeTo.X && nodePDom.Y < nodeTo.Y) {
        from = postDom;
        continue;
      }
    }

    // possible false positive, so check recursively..
    for (const size_t succ : nodeFrom.successors) {
      if (succ == to) {
        return true;
      }
      auto &nodeSucc = graph[succ];
      if (nodeSucc.X < nodeTo.X && nodeSucc.Y < nodeTo.Y) {
        if (visited.insert(succ).second) {
          worklist.push_back(succ);
        }
      }
    }
    if (worklist.empty()) {
      return false;
    }
    from = worklist.back();
    worklist.pop_back();
  }
  return false;
}

bool Reachability::isReachable(BasicBlock *from, BasicBlock *to) const {
  auto fromI = indexMap.find(from);
  if (fromI == indexMap.end()) {
    return false;
  }

  auto toI = indexMap.find(to);
  if (toI == indexMap.end()) {
    return false;
  }

  return from == to || isReachableImpl(fromI->second, toI->second);
}

}  // namespace vecz
