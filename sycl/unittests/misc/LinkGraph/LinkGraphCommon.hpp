//==---- LinkGraphCommon.cpp --- link_graph uniitest common helpers --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/link_graph.hpp>
#include <gtest/gtest.h>

template <typename T>
class IntrospectiveLinkGraph : public sycl::detail::LinkGraph<T> {
public:
  using sycl::detail::LinkGraph<T>::LinkGraph;
  using Node = typename sycl::detail::LinkGraph<T>::Node;

  IntrospectiveLinkGraph(sycl::detail::LinkGraph<T> &&Graph)
      : sycl::detail::LinkGraph<T>(std::move(Graph)) {}

  std::optional<const Node *> FindNode(const T &Value) const {
    auto It = std::find_if(this->Nodes.begin(), this->Nodes.end(),
                           [&](const Node &N) { return N.Value == Value; });
    if (It == this->Nodes.end())
      return std::nullopt;
    return &(*It);
  }

  std::optional<size_t> GetNodeLocation(const T &Value) const {
    for (size_t I = 0; I < this->Nodes.size(); ++I)
      if (this->Nodes[I].Value == Value)
        return I;
    return std::nullopt;
  }

  void AssertNodeAlive(const T &Value, const std::vector<T> &ExpectedParents,
                       const std::vector<T> &ExpectedChildren) const {
    std::optional<const Node *> N = FindNode(Value);
    ASSERT_TRUE(N.has_value()) << " for " << Value;
    ASSERT_TRUE((*N)->Alive) << " for " << Value;
    ASSERT_EQ((*N)->Parents.size(), ExpectedParents.size()) << " for " << Value;
    ASSERT_EQ((*N)->Children.size(), ExpectedChildren.size())
        << " for " << Value;
    for (const T &ExpectedParent : ExpectedParents) {
      std::optional<size_t> ParentLocation = GetNodeLocation(ExpectedParent);
      ASSERT_TRUE(ParentLocation.has_value());
      auto FoundParent = std::find((*N)->Parents.begin(), (*N)->Parents.end(),
                                   *ParentLocation);
      ASSERT_NE(FoundParent, (*N)->Parents.end()) << " for " << Value;
    }
    for (const T &ExpectedChild : ExpectedChildren) {
      std::optional<size_t> ChildLocation = GetNodeLocation(ExpectedChild);
      ASSERT_TRUE(ChildLocation.has_value());
      auto FoundChild = std::find((*N)->Children.begin(), (*N)->Children.end(),
                                  ChildLocation);
      ASSERT_NE(FoundChild, (*N)->Children.end()) << " for " << Value;
    }
  }

  void AssertNodeDead(const T &Value) const {
    std::optional<const Node *> N = FindNode(Value);
    ASSERT_TRUE(N.has_value()) << " for " << Value;
    ASSERT_FALSE((*N)->Alive) << " for " << Value;
  }

  void AssertAliveValues(const std::vector<T> &ExpectedValues) const {
    std::vector<T> AliveValues = this->GetNodeValues();
    std::vector<T> ExpectedValuesCopy = ExpectedValues;
    std::sort(AliveValues.begin(), AliveValues.end());
    std::sort(ExpectedValuesCopy.begin(), ExpectedValuesCopy.end());

    ASSERT_EQ(AliveValues.size(), ExpectedValuesCopy.size());
    ASSERT_EQ(AliveValues, ExpectedValuesCopy);
  }
};

// Graph 1:
//    A
//    ↓
//    B
const IntrospectiveLinkGraph<char> Graph1(std::vector<char>{'A', 'B'},
                                          std::vector<std::vector<size_t>>{{1},
                                                                           {}});

// Graph 2:
//    A
//    ↓
//    B
//    ↓
//    C
const IntrospectiveLinkGraph<char> Graph2(std::vector<char>{'A', 'B', 'C'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {2}, {}});

// Graph 3:
//    A
//   ↙ ↘
//  B   C
const IntrospectiveLinkGraph<char> Graph3(std::vector<char>{'A', 'B', 'C'},
                                          std::vector<std::vector<size_t>>{
                                              {1, 2}, {}, {}});

// Graph 4:
//    A    D
//   ↙ ↘ ↙
//  B   C
const IntrospectiveLinkGraph<char> Graph4(std::vector<char>{'A', 'B', 'C', 'D'},
                                          std::vector<std::vector<size_t>>{
                                              {1, 2}, {}, {}, {2}});

// Graph 5:
//    A
//    ↕
//    B
const IntrospectiveLinkGraph<char> Graph5(std::vector<char>{'A', 'B'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {0}});

// Graph 6:
//    A
//   ↙ ↖
//  B → C
const IntrospectiveLinkGraph<char> Graph6(std::vector<char>{'A', 'B', 'C'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {2}, {0}});

// Graph 7:
//    A
//    ↓
//    B
//   ↙ ↖
//  C → D
const IntrospectiveLinkGraph<char> Graph7(std::vector<char>{'A', 'B', 'C', 'D'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {2}, {3}, {1}});

// Graph 8:
//    A
//   ↙ ↖
//  B → C
//  ↓
//  D
const IntrospectiveLinkGraph<char> Graph8(std::vector<char>{'A', 'B', 'C', 'D'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {2, 3}, {0}, {}});

// Graph 9:
//    A  C
//    ↓  ↓
//    B  D
const IntrospectiveLinkGraph<char> Graph9(std::vector<char>{'A', 'B', 'C', 'D'},
                                          std::vector<std::vector<size_t>>{
                                              {1}, {}, {3}, {}});
