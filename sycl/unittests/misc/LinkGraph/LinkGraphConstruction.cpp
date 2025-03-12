//==---- LinkGraphConstruction.cpp --- link_graph construction unit test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinkGraphCommon.hpp"

TEST(LinkGraph1ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph1.Clone();
  Graph.AssertNodeAlive('A', {}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {});
  Graph.AssertAliveValues({'A', 'B'});
}

TEST(LinkGraph2ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph2.Clone();
  Graph.AssertNodeAlive('A', {}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {'C'});
  Graph.AssertNodeAlive('C', {'B'}, {});
  Graph.AssertAliveValues({'A', 'B', 'C'});
}

TEST(LinkGraph3ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph3.Clone();
  Graph.AssertNodeAlive('A', {}, {'B', 'C'});
  Graph.AssertNodeAlive('B', {'A'}, {});
  Graph.AssertNodeAlive('C', {'A'}, {});
  Graph.AssertAliveValues({'A', 'B', 'C'});
}

TEST(LinkGraph4ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph4.Clone();
  Graph.AssertNodeAlive('A', {}, {'B', 'C'});
  Graph.AssertNodeAlive('B', {'A'}, {});
  Graph.AssertNodeAlive('C', {'A', 'D'}, {});
  Graph.AssertNodeAlive('D', {}, {'C'});
  Graph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph5ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph5.Clone();
  Graph.AssertNodeAlive('A', {'B'}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {'A'});
  Graph.AssertAliveValues({'A', 'B'});
}

TEST(LinkGraph6ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph6.Clone();
  Graph.AssertNodeAlive('A', {'C'}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {'C'});
  Graph.AssertNodeAlive('C', {'B'}, {'A'});
  Graph.AssertAliveValues({'A', 'B', 'C'});
}

TEST(LinkGraph7ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph7.Clone();
  Graph.AssertNodeAlive('A', {}, {'B'});
  Graph.AssertNodeAlive('B', {'A', 'D'}, {'C'});
  Graph.AssertNodeAlive('C', {'B'}, {'D'});
  Graph.AssertNodeAlive('D', {'C'}, {'B'});
  Graph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph8ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph8.Clone();
  Graph.AssertNodeAlive('A', {'C'}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {'C', 'D'});
  Graph.AssertNodeAlive('C', {'B'}, {'A'});
  Graph.AssertNodeAlive('D', {'B'}, {});
  Graph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph9ConstructionTest, LinkGraphConstructionTest) {
  IntrospectiveLinkGraph<char> Graph = Graph9.Clone();
  Graph.AssertNodeAlive('A', {}, {'B'});
  Graph.AssertNodeAlive('B', {'A'}, {});
  Graph.AssertNodeAlive('C', {}, {'D'});
  Graph.AssertNodeAlive('D', {'C'}, {});
  Graph.AssertAliveValues({'A', 'B', 'C', 'D'});
}
