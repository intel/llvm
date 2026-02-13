//==------- LinkGraphPoisoning.cpp --- link_graph poisoning unit test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinkGraphCommon.hpp"

TEST(LinkGraph1PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph1.Clone();

  // Poison A, which should make B a root.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {});
  Graph.AssertAliveValues({'B'});
}

TEST(LinkGraph1PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph1.Clone();

  // Poison B, which should kill A.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph2PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph2.Clone();

  // Poison A, which should make B a root.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {'C'});
  Graph.AssertNodeAlive('C', {'B'}, {});
  Graph.AssertAliveValues({'B', 'C'});
}

TEST(LinkGraph2PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph2.Clone();

  // Poison B, which should kill A and make C a root.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeAlive('C', {}, {});
  Graph.AssertAliveValues({'C'});
}

TEST(LinkGraph2PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph2.Clone();

  // Poison C, which should kill A and B.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph3PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph3.Clone();

  // Poison A, which should make B and C roots.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {});
  Graph.AssertNodeAlive('C', {}, {});
  Graph.AssertAliveValues({'B', 'C'});
}

TEST(LinkGraph3PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph3.Clone();

  // Poison B, which should kill A and make C a root.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeAlive('C', {}, {});
  Graph.AssertAliveValues({'C'});
}

TEST(LinkGraph3PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph3.Clone();

  // Poison C, which should kill A and make B a root.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {});
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({'B'});
}

TEST(LinkGraph3PoisonTest4, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph3.Clone();

  // Poison both B and C, which should kill A.
  Graph.Poison([](const char &C) { return C == 'B' || C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph4PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph4.Clone();

  // Poison A, making B a root and leaving C as a child of D.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {});
  Graph.AssertNodeAlive('C', {'D'}, {});
  Graph.AssertNodeAlive('D', {}, {'C'});
  Graph.AssertAliveValues({'B', 'C', 'D'});
}

TEST(LinkGraph4PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph4.Clone();

  // Poison B, killing A and leaving C as a child of D.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeAlive('C', {'D'}, {});
  Graph.AssertNodeAlive('D', {}, {'C'});
  Graph.AssertAliveValues({'C', 'D'});
}

TEST(LinkGraph4PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph4.Clone();

  // Poison C, killing A and D leaving B as a root.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {}, {});
  Graph.AssertNodeDead('C');
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({'B'});
}

TEST(LinkGraph4PoisonTest4, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph4.Clone();

  // Poison D, leaving A as the root and sole parent to both B and C.
  Graph.Poison([](const char &C) { return C == 'D'; });

  Graph.AssertNodeAlive('A', {}, {'B', 'C'});
  Graph.AssertNodeAlive('B', {'A'}, {});
  Graph.AssertNodeAlive('C', {'A'}, {});
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({'A', 'B', 'C'});
}

TEST(LinkGraph5PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph5.Clone();

  // Poison A, killing B.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph5PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph5.Clone();

  // Poison B, killing A.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph6PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph6.Clone();

  // Poison A, killing B and C.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph6PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph6.Clone();

  // Poison B, killing A and C.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph6PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph6.Clone();

  // Poison C, killing A and B.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph7PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph7.Clone();

  // Poison A, leaving the cycle between B, C and D.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeAlive('B', {'D'}, {'C'});
  Graph.AssertNodeAlive('C', {'B'}, {'D'});
  Graph.AssertNodeAlive('D', {'C'}, {'B'});
  Graph.AssertAliveValues({'B', 'C', 'D'});
}

TEST(LinkGraph7PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph7.Clone();

  // Poison B, killing A, C and D.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph7PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph7.Clone();

  // Poison C, killing A, B and D.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph7PoisonTest4, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph7.Clone();

  // Poison D, killing A, B and C.
  Graph.Poison([](const char &C) { return C == 'D'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({});
}

TEST(LinkGraph8PoisonTest1, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph8.Clone();

  // Poison A, killing B and C, leaving D as a root.
  Graph.Poison([](const char &C) { return C == 'A'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeAlive('D', {}, {});
  Graph.AssertAliveValues({'D'});
}

TEST(LinkGraph8PoisonTest2, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph8.Clone();

  // Poison B, killing A and C, leaving D as a root.
  Graph.Poison([](const char &C) { return C == 'B'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeAlive('D', {}, {});
  Graph.AssertAliveValues({'D'});
}

TEST(LinkGraph8PoisonTest3, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph8.Clone();

  // Poison C, killing A and B, leaving D as a root.
  Graph.Poison([](const char &C) { return C == 'C'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeAlive('D', {}, {});
  Graph.AssertAliveValues({'D'});
}

TEST(LinkGraph8PoisonTest4, LinkGraphPoisoningTest) {
  IntrospectiveLinkGraph<char> Graph = Graph8.Clone();

  // Poison D, killing A, B and C.
  Graph.Poison([](const char &C) { return C == 'D'; });

  Graph.AssertNodeDead('A');
  Graph.AssertNodeDead('B');
  Graph.AssertNodeDead('C');
  Graph.AssertNodeDead('D');
  Graph.AssertAliveValues({});
}
