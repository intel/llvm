//==--------- LinkGraphUnify.cpp --- link_graph unifying unit test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinkGraphCommon.hpp"

using namespace std::literals;

TEST(LinkGraph1Unify1, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph1.Clone());
  Graphs.emplace("Tag2"s, Graph1.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup{"Tag1"s, "Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph{std::move(UnifiedGraphs[TagGroup])};
  UnifiedGraph.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph.AssertAliveValues({'A', 'B'});
}

TEST(LinkGraph1Unify2, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph1.Clone());
  Graphs.emplace("Tag2"s, Graph1.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'A'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph1.AssertAliveValues({'A', 'B'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('B', {}, {});
  UnifiedGraph2.AssertAliveValues({'B'});
}

TEST(LinkGraph1Unify3, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph1.Clone());
  Graphs["Tag1"s].Poison([](const char &C) { return C == 'A'; });
  Graphs.emplace("Tag2"s, Graph1.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('B', {}, {});
  UnifiedGraph1.AssertAliveValues({'B'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph2.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph2.AssertAliveValues({'A', 'B'});
}

TEST(LinkGraph1Unify4, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph1.Clone());
  Graphs.emplace("Tag2"s, Graph1.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'B'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph1.AssertAliveValues({'A', 'B'});

  // Tag2 should not have a group as it has no trees.
}

TEST(LinkGraph1Unify5, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph1.Clone());
  Graphs.emplace("Tag2"s, Graph1.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'A'; });
  Graphs.emplace("Tag3"s, Graph1.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s, "Tag3"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph1.AssertAliveValues({'A', 'B'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('B', {}, {});
  UnifiedGraph2.AssertAliveValues({'B'});
}

TEST(LinkGraph7Unify1, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph7.Clone());
  Graphs.emplace("Tag2"s, Graph7.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup{"Tag1"s, "Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph{std::move(UnifiedGraphs[TagGroup])};
  UnifiedGraph.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph.AssertNodeAlive('B', {'A', 'D'}, {'C'});
  UnifiedGraph.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph7Unify2, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph7.Clone());
  Graphs.emplace("Tag2"s, Graph7.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'A'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A', 'D'}, {'C'});
  UnifiedGraph1.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph1.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph1.AssertAliveValues({'A', 'B', 'C', 'D'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('B', {'D'}, {'C'});
  UnifiedGraph2.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph2.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph2.AssertAliveValues({'B', 'C', 'D'});
}

TEST(LinkGraph7Unify3, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph7.Clone());
  Graphs["Tag1"s].Poison([](const char &C) { return C == 'A'; });
  Graphs.emplace("Tag2"s, Graph7.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('B', {'D'}, {'C'});
  UnifiedGraph1.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph1.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph1.AssertAliveValues({'B', 'C', 'D'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph2.AssertNodeAlive('B', {'A', 'D'}, {'C'});
  UnifiedGraph2.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph2.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph2.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph7Unify4, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph7.Clone());
  Graphs["Tag1"s].Poison([](const char &C) { return C == 'B'; });
  Graphs.emplace("Tag2"s, Graph7.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'A'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph{std::move(UnifiedGraphs[TagGroup])};
  UnifiedGraph.AssertNodeAlive('B', {'D'}, {'C'});
  UnifiedGraph.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph.AssertAliveValues({'B', 'C', 'D'});
}

TEST(LinkGraph7Unify5, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph7.Clone());
  Graphs.emplace("Tag2"s, Graph7.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'B'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph{std::move(UnifiedGraphs[TagGroup])};
  UnifiedGraph.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph.AssertNodeAlive('B', {'A', 'D'}, {'C'});
  UnifiedGraph.AssertNodeAlive('C', {'B'}, {'D'});
  UnifiedGraph.AssertNodeAlive('D', {'C'}, {'B'});
  UnifiedGraph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph9Unify1, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph9.Clone());
  Graphs.emplace("Tag2"s, Graph9.Clone());
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{1});

  std::vector<std::string> TagGroup{"Tag1"s, "Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph{std::move(UnifiedGraphs[TagGroup])};
  UnifiedGraph.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph.AssertNodeAlive('C', {}, {'D'});
  UnifiedGraph.AssertNodeAlive('D', {'C'}, {});
  UnifiedGraph.AssertAliveValues({'A', 'B', 'C', 'D'});
}

TEST(LinkGraph9Unify2, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph9.Clone());
  Graphs.emplace("Tag2"s, Graph9.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'A'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{3});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph1.AssertAliveValues({'A', 'B'});

  std::vector<std::string> TagGroup2{"Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('B', {}, {});
  UnifiedGraph2.AssertAliveValues({'B'});

  std::vector<std::string> TagGroup3{"Tag1"s, "Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph3{
      std::move(UnifiedGraphs[TagGroup3])};
  UnifiedGraph3.AssertNodeAlive('C', {}, {'D'});
  UnifiedGraph3.AssertNodeAlive('D', {'C'}, {});
  UnifiedGraph3.AssertAliveValues({'C', 'D'});
}

TEST(LinkGraph9Unify3, LinkGraphUnifyTest) {
  std::map<std::string, sycl::detail::LinkGraph<char>> Graphs;
  Graphs.emplace("Tag1"s, Graph9.Clone());
  Graphs.emplace("Tag2"s, Graph9.Clone());
  Graphs["Tag2"s].Poison([](const char &C) { return C == 'B'; });
  std::map<std::vector<std::string>, sycl::detail::LinkGraph<char>>
      UnifiedGraphs = sycl::detail::UnifyGraphs(Graphs);

  ASSERT_EQ(UnifiedGraphs.size(), size_t{2});

  std::vector<std::string> TagGroup1{"Tag1"s};
  IntrospectiveLinkGraph<char> UnifiedGraph1{
      std::move(UnifiedGraphs[TagGroup1])};
  UnifiedGraph1.AssertNodeAlive('A', {}, {'B'});
  UnifiedGraph1.AssertNodeAlive('B', {'A'}, {});
  UnifiedGraph1.AssertAliveValues({'A', 'B'});

  std::vector<std::string> TagGroup2{"Tag1"s, "Tag2"s};
  IntrospectiveLinkGraph<char> UnifiedGraph2{
      std::move(UnifiedGraphs[TagGroup2])};
  UnifiedGraph2.AssertNodeAlive('C', {}, {'D'});
  UnifiedGraph2.AssertNodeAlive('D', {'C'}, {});
  UnifiedGraph2.AssertAliveValues({'C', 'D'});
}
