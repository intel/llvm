//==----- link_graph.hpp - Graph utility for resolving linking groups ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header contains an implementation of a "link graph" utility intended to
// be used by the SYCL runtime for resolving binary device binary groupings
// for images, based on whether they are supported by some of the devices in the
// linking operations.
//
// The link graphs (represented by LinkGraph) consist of one or more trees of
// nodes, each holding a value of a type T. These nodes can have zero or more
// parents and zero or more children. In the way this class is used by SYCL,
// parents represent device binaries that imports symbol from the device binary
// represented by the node, and children represent device binaries exporting
// symbols imported by the device binary represented by the node.
// Note that trees can contain cycles.
// In these trees, roots are nodes with no parents and leaves are nodes with no
// children. Note that a tree can have zero or more roots and zero or more
// leaves, zero in both cases only being possible when cycles exist in the tree.
//
// To allow for grouping devices by the binaries they support, the following
// operations are supported on a link graph:
// * Poisoning: By poisoning a graph, nodes that do not satisfy a given
//              requirement will be removed from the graph. Additionally,
//              poisoning propagates up the tree, killing all parents of
//              poisoned nodes. By poisoning nodes, children will lose the node
//              as a parent and may become roots (but never leaves.)
// * Unifying: Unifying graphs attempts to find common trees in a collection of
//             graphs, resulting in a potentially new set of graphs containing
//             all full common trees of a sub-collection of the graphs. Note
//             that sub-trees do not count as full common trees.
//
// Poisoning example:
//
//      A    D    Poison B     D
//     ↙ ↘ ↙    ---------->   ↓
//    B   C                    C
//
// Unify example:
//
//       Graph 1         Graph 2                  {1}         {2}   {1 & 2}
//   [   A    D  E ]    [ D   E ]    Unify    [   A    D ]   [ D ]   [ E ]
//   [  ↙ ↘ ↙   ↓  ] & [ ↓   ↓  ]  ------->  [  ↙ ↘ ↙   ] & [ ↓ ] & [ ↓ ]
//   [ B   C     F ]    [ C   F ]             [ B   C    ]   [ C ]   [ F ]
//
//   NOTE: Though D and C are in both {1} and {2}, they are not unified as their
//         full trees are not the same.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Representation of the link graph.
template <typename T> class LinkGraph {
protected:
  // Representation of a node in the graph.
  struct Node {
    Node(const T &Val) : Value(Val) {}

    T Value;
    bool Alive = true;
    // Parents and children are represented by the corresponding offsets into
    // the node list. Since nodes in a node list never move, this simplifies
    // copying the lists.
    std::vector<size_t> Parents, Children;
  };

public:
  LinkGraph() = default;
  LinkGraph(LinkGraph &&Other) = default;

  LinkGraph(const std::vector<T> &Values,
            const std::vector<std::vector<size_t>> &Links) {
    assert(Values.size() == Links.size());

    Nodes.reserve(Values.size());
    for (const T &Value : Values)
      Nodes.emplace_back(Value);

    // Create links after the nodes have all been materialized.
    for (size_t I = 0; I < Values.size(); ++I) {
      Node &N = Nodes[I];
      N.Children = Links[I];
      for (size_t ChildIndex : N.Children) {
        assert(ChildIndex < Nodes.size());
        Nodes[ChildIndex].Parents.push_back(I);
      }
    }
  }

  LinkGraph Clone() const { return LinkGraph{*this}; }

  /// Poisons the nodes in the graph based on a poisoning condition. Poisoning
  /// a node causes the node to be removed from the graph will recursively
  /// poison the parents of the node, even if they do not satisfy the poisoning
  /// condition.
  /// \param PoisonCondFunc is the poisoning condition.
  template <typename PoisonCondFuncT>
  void Poison(const PoisonCondFuncT &PoisonCondFunc) {
    // Empty graph, so no work to be done.
    if (Nodes.empty())
      return;

    // Keep track of seen nodes and how often they have been seen.
    std::vector<uint32_t> Seen(Nodes.size(), 0);

    // Keep a growing work-list of nodes to check. We should get through all
    // nodes this way, excluding dead nodes.
    std::vector<size_t> WorkList;
    WorkList.reserve(Nodes.size());

    // Pass through nodes to count dead nodes and add all leaves to the
    // work-list.
    size_t InitialDeadNodeCount = 0;
    for (size_t I = 0; I < Nodes.size(); ++I) {
      const Node &CurrentNode = Nodes[I];
      if (!CurrentNode.Alive) {
        Seen[I] = 1;
        ++InitialDeadNodeCount;
      } else if (CurrentNode.Children.empty()) {
        Seen[I] = 1;
        WorkList.push_back(I);
      }
    }

    // Poison all leaves.
    PoisonInner(PoisonCondFunc, WorkList, Seen);

    // If the worklist doesn't have all live nodes, there must be cycles in the
    // list. We handle these by finding any part of them that we haven't yet
    // seen and process them in separation until there is no more work to do.
    while (WorkList.size() < Nodes.size() - InitialDeadNodeCount) {
      const size_t PreviousWorkListSize = WorkList.size();

      // Find the first unseen node. This will be the basis of our new work.
      for (size_t I = 0; I < Nodes.size(); ++I) {
        // Processed nodes will all have been seen by all children, or if they
        // are leaves or dead they may have been seen more often.
        if (Seen[I] >= Nodes[I].Children.size())
          continue;
        // If we pick an unseen node, we set its seen counter to the number of
        // children to prevent the children from re-adding it to the work-list.
        Seen[I] = Nodes[I].Children.size();

        WorkList.push_back(I);
        // It should not be possible for an unseen node to be dead at this
        // point.
        assert(Nodes[I].Alive);
        break;
      }

      // The amount of work must have grown after looking for cycles, otherwise
      // something is very wrong.
      assert(PreviousWorkListSize < WorkList.size());

      // Poison the found cycle, skipping all old work.
      PoisonInner(PoisonCondFunc, WorkList, Seen,
                  /*StartIndex=*/PreviousWorkListSize);
    }
  }

  /// Gets the values of all alive nodes in the graph.
  /// \returns the values of all alive nodes in the graph.
  std::vector<T> GetNodeValues() const noexcept {
    std::vector<T> NodeValues;
    NodeValues.reserve(std::count_if(Nodes.begin(), Nodes.end(),
                                     [](const Node &N) { return N.Alive; }));
    for (const Node &N : Nodes)
      if (N.Alive)
        NodeValues.emplace_back(N.Value);
    return NodeValues;
  }

protected:
  // Copy ctor is private to avoid unintentional expensive copies. The Clone()
  // member function should be used instead.
  LinkGraph(const LinkGraph &Other) = default;

  /// Helper to run the inner poisoning logic. We may need to run it multiple
  /// times to handle cycles in the graph.
  /// \param PoisonCondFunc is the poisoning condition.
  /// \param WorkList is a reference to a vector of nodes to visit. This may
  ///        grow during the execution of the function.
  /// \param Seen is a reference to counters for how often a given node has been
  ///        observed by child. These match 1:1 with the nodes in the graph.
  /// \param StartIndex is the index into the work list to start from.
  template <typename PoisonCondFuncT>
  void PoisonInner(const PoisonCondFuncT &PoisonCondFunc,
                   std::vector<size_t> &WorkList, std::vector<uint32_t> &Seen,
                   size_t StartIndex = 0) {
    // Go through the work-list and kill poisoned nodes.
    // Note that the work-list may grow as we go through it.
    for (size_t I = StartIndex; I < WorkList.size(); ++I) {
      const size_t CurrentNodeIndex = WorkList[I];
      Node &CurrentNode = Nodes[CurrentNodeIndex];

      // If the node isn't already poisoned, check condition.
      if (CurrentNode.Alive)
        CurrentNode.Alive = !PoisonCondFunc(CurrentNode.Value);

      // For each parent:
      //  1. If this node was poisoned by the condition or had already
      //     been poisoned, poison the parent.
      //  2. If this node poisoned the parent or if this is the last child
      //     to see the parent, add the parent to the work-list.
      for (size_t ParentIndex : CurrentNode.Parents) {
        assert(ParentIndex < Nodes.size());
        Node &ParentNode = Nodes[ParentIndex];
        ++Seen[ParentIndex];

        // If the parent is already dead, the node must already be in the
        // work list and may already have been processed.
        if (!ParentNode.Alive)
          continue;
        assert(!ParentNode.Children.empty());

        // 1. Propagate poisoning.
        if (!CurrentNode.Alive)
          ParentNode.Alive = false;

        // 2. Add to work-list.
        // NOTE: We intentionally check direct equality between number of
        //       children and the seen counter. This lets us avoid cases where
        //       the parent was added to the work-list from elsewhere, such as
        //       from cycle discovery.
        if (!CurrentNode.Alive ||
            ParentNode.Children.size() == Seen[ParentIndex])
          WorkList.push_back(ParentIndex);
      }

      // If the node is dead, clear its relations. Note that it must have
      // poisoned its parents, so it does not need to remove itself from
      // the children lists as they will inevitably be cleared.
      if (!CurrentNode.Alive) {
        for (size_t ChildIndex : CurrentNode.Children) {
          assert(ChildIndex < Nodes.size());
          Node &Child = Nodes[ChildIndex];
          // Dead children must be skipped too.
          if (!Child.Alive)
            continue;
          auto Pos = std::find(Child.Parents.begin(), Child.Parents.end(),
                               CurrentNodeIndex);
          assert(Pos != Child.Parents.end());
          Child.Parents.erase(Pos);
        }
        CurrentNode.Parents.clear();
        CurrentNode.Children.clear();
      }
    }
  }

  /// Copies a collection of nodes, adapting their dependency offsets.
  /// Requires that the collection is self-contained, i.e. that all dependencies
  /// of nodes in the collection are also in the collection.
  /// \param AdoptiveNodes is a map between indices and the node to copy.
  void AdoptNodes(const std::map<size_t, const Node &> &AdoptiveNodes) {
    const size_t BaseOffset = Nodes.size();
    Nodes.reserve(AdoptiveNodes.size());

    // Lambda for updating an index to its new location in Nodes.
    // Note that AdoptiveNodes is sorted by its keys, so the updated indices are
    // guaranteed to be sorted after the update too.
    auto UpdateIndex = [BaseOffset, &AdoptiveNodes](size_t &I) {
      auto InAdoptives = AdoptiveNodes.find(I);
      assert(InAdoptives != AdoptiveNodes.end());
      I = BaseOffset + std::distance(AdoptiveNodes.begin(), InAdoptives);
    };

    for (const auto &N : AdoptiveNodes) {
      Node &NewNode = Nodes.emplace_back(N.second);
      for (size_t &ParentIndex : NewNode.Parents)
        UpdateIndex(ParentIndex);
      for (size_t &ChildIndex : NewNode.Children)
        UpdateIndex(ChildIndex);
    }
  }

  /// Traverses the tree, starting from a given index, recording the nodes.
  /// \param Graphs are the graphs to find common trees for.
  /// \param NodeIndex is the node to start the traversal from.
  /// \param Visited is a reference to a vector marking whether nodes have
  ///        previously been analyzed.
  /// \param MappedTree is a reference to a vector to record the nodes into.
  template <typename TagT>
  static void MapTree(const std::map<TagT, LinkGraph> &Graphs, size_t NodeIndex,
                      std::vector<bool> &Visited,
                      std::vector<size_t> &MappedTree) {
    assert(NodeIndex < Visited.size());
    if (Visited[NodeIndex])
      return;
    Visited[NodeIndex] = true;

    MappedTree.push_back(NodeIndex);

    for (const auto &Graph : Graphs) {
      const auto &Node = Graph.second.Nodes[NodeIndex];
      // Map both parents and children.
      for (size_t ParentIndex : Node.Parents)
        MapTree(Graphs, ParentIndex, Visited, MappedTree);
      for (size_t ChildIndex : Node.Children)
        MapTree(Graphs, ChildIndex, Visited, MappedTree);
    }
  }

  /// Helper function for finding common trees in graphs, searching around a
  /// given node.
  /// \param Graphs are the graphs to find common trees for.
  /// \param NodeIndex is the node to start the search from.
  /// \param Visited is a reference to a vector marking whether nodes have
  ///        previously been analyzed.
  /// \returns a map between tag groups and their common sub-trees, represented
  ///          by maps of indices of nodes in the graphs and references to one
  ///          variant of the corresponding node.
  template <typename TagT>
  static std::map<std::vector<TagT>, std::map<size_t, const Node &>>
  FindCommonTrees(const std::map<TagT, LinkGraph> &Graphs, size_t NodeIndex,
                  std::vector<bool> &Visited) {
    std::map<std::vector<TagT>, std::map<size_t, const Node &>> Result;

    // Map out the tree and register the visited nodes.
    std::vector<size_t> Tree;
    MapTree(Graphs, NodeIndex, Visited, Tree);

    // To group tags, we create a matrix of which tags use the tree nodes.
    std::vector<std::pair<TagT, std::vector<bool>>> NodeUseMatrix;
    NodeUseMatrix.reserve(Graphs.size());
    for (const auto &Graph : Graphs) {
      auto &Row = NodeUseMatrix.emplace_back(Graph.first, std::vector<bool>{});
      Row.second.reserve(Tree.size());
      for (size_t TreeNodeIndex : Tree)
        Row.second.push_back(Graph.second.Nodes[TreeNodeIndex].Alive);
    }

    // Sorting the matrix lexicographically makes it easy to iteratively find
    // the tag groupings.
    std::sort(NodeUseMatrix.begin(), NodeUseMatrix.end(),
              [](const std::pair<TagT, std::vector<bool>> &LHS,
                 const std::pair<TagT, std::vector<bool>> &RHS) {
                return LHS.second > RHS.second;
              });

    // After sorting, the end may have tags that do not use any of the nodes. We
    // can safely drop these.
    auto FirstEmpty = NodeUseMatrix.end();
    while (FirstEmpty != NodeUseMatrix.begin()) {
      if (std::any_of((FirstEmpty - 1)->second.begin(),
                      (FirstEmpty - 1)->second.end(), [](bool X) { return X; }))
        break;
      --FirstEmpty;
    }
    NodeUseMatrix.erase(FirstEmpty, NodeUseMatrix.end());

    // Now we can group tags together by their node usage and create their
    // common tree mappings.
    std::vector<TagT> CurrentTagGroup;
    for (size_t I = 0; I < NodeUseMatrix.size(); ++I) {
      const std::pair<TagT, std::vector<bool>> &Row = NodeUseMatrix[I];
      CurrentTagGroup.push_back(Row.first);

      // If the next row has the same usage, group them together and defer
      // processing.
      if (I + 1 < NodeUseMatrix.size() &&
          NodeUseMatrix[I + 1].second == Row.second)
        continue;

      // Construct tree map for the current group.
      const LinkGraph &Graph = Graphs.at(Row.first);
      std::map<size_t, const Node &> TreeMap;
      for (size_t TreeNodeIndex : Tree)
        TreeMap.emplace(TreeNodeIndex, Graph.Nodes[TreeNodeIndex]);
      Result.emplace(std::move(CurrentTagGroup), std::move(TreeMap));

      // Clear the tag group for the next group.
      CurrentTagGroup.clear();
    }

    return Result;
  }

  // Nodes in the graph.
  std::vector<Node> Nodes;

  template <typename TagT, typename GraphValT>
  friend std::map<std::vector<TagT>, LinkGraph<GraphValT>>
  UnifyGraphs(const std::map<TagT, LinkGraph<GraphValT>> &);
};

/// Unifies graphs to find common trees for a given tag, trying to reduce the
/// sets of common trees.
/// The graphs are required to contain the same nodes in the same order. The
/// dependencies and the state of the nodes may differ. The resulting graphs do
/// not make the same guarantees.
/// \param Graphs are the graphs to unify.
/// \returns a map between groupings of tags and graphs containing their common
///          trees.
template <typename TagT, typename GraphValT>
inline std::map<std::vector<TagT>, LinkGraph<GraphValT>>
UnifyGraphs(const std::map<TagT, LinkGraph<GraphValT>> &Graphs) {
  std::map<std::vector<TagT>, LinkGraph<GraphValT>> Results;
  if (Graphs.empty())
    return Results;

  const size_t CommonSize = Graphs.begin()->second.Nodes.size();

  // Due to the requirement that the graphs must contain the same nodes, we make
  // a simple check for the sizes.
  assert(std::all_of(Graphs.begin(), Graphs.end(),
                     [CommonSize](const auto &Graph) {
                       return Graph.second.Nodes.size() == CommonSize;
                     }));

  std::vector<bool> Visited(CommonSize, false);
  for (size_t I = 0; I < CommonSize; ++I) {
    if (Visited[I])
      continue;

    // Find the common trees around the trees based around the current node.
    auto CommonTrees =
        LinkGraph<GraphValT>::FindCommonTrees(Graphs, I, Visited);

    // Adopt the common trees into the result graphs.
    for (const auto &CommonTreesGroup : CommonTrees)
      Results[CommonTreesGroup.first].AdoptNodes(CommonTreesGroup.second);
  }

  return Results;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
