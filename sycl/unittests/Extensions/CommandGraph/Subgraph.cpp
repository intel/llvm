//==--------------------------- Subgraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, SubGraph) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->MNodeType ==
              experimental::node_type::subgraph);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph on
  // finalization. we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->MNodeType == experimental::node_type::subgraph);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 4 nodes: the two nodes from the subgraph are
  // merged into the main graph in place of the subgraph node.
  ASSERT_EQ(Schedule.size(), 4ul);
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node1MainGraph)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node3MainGraph)));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, SubGraphWithEmptyNode) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Empty1Graph =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on(Node1Graph)});
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Empty1Graph)});

  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->MNodeType ==
              experimental::node_type::subgraph);
  // Check the structure of the main graph.
  // 1 root connected to 1 successor (the single root of the subgraph)
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph.
  // we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->MNodeType == experimental::node_type::subgraph);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 5 nodes: 2 main graph nodes and 3 subgraph
  // nodes which have been merged.
  ASSERT_EQ(Schedule.size(), 5ul);
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node1MainGraph)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty()); // empty node inside the subgraph
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node3MainGraph)));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, SubGraphWithEmptyNodeLast) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto EmptyGraph =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on(Node2Graph)});

  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->MNodeType ==
              experimental::node_type::subgraph);
  // Check the structure of the main graph.
  // 1 root connected to 1 successor (the single root of the subgraph)
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  // Subgraph nodes are duplicated when inserted to parent graph.
  // we thus check the node content only.
  const bool CompareContentOnly = true;
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1MainGraph)
                  ->MSuccessors.front()
                  .lock()
                  ->MNodeType == experimental::node_type::subgraph);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MPredecessors.size(),
            0lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2MainGraph)->MPredecessors.size(),
            1lu);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // The schedule list must contain 5 nodes: 2 main graph nodes and 3 subgraph
  // nodes which have been merged.
  ASSERT_EQ(Schedule.size(), 5ul);
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node1MainGraph)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node1Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)
                  ->isSimilar(sycl::detail::getSyclObjImpl(Node2Graph),
                              CompareContentOnly));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty()); // empty node inside the subgraph
  ScheduleIt++;
  ASSERT_TRUE(
      (*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node3MainGraph)));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, RecordSubGraph) {
  // Record sub-graph with two nodes
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node1Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });
  Graph.end_recording(Queue);
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  MainGraph.begin_recording(Queue);
  auto Node1MainGraph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2MainGraph = Queue.submit([&](handler &cgh) {
    cgh.depends_on(Node1MainGraph);
    cgh.ext_oneapi_graph(GraphExec);
  });
  auto Node3MainGraph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node2MainGraph);
    cgh.single_task<TestKernel<>>([]() {});
  });
  MainGraph.end_recording(Queue);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();

  // The schedule list must contain 4 nodes: 2 main graph nodes and 2 subgraph
  // nodes which have been merged in to the main graph.
  ASSERT_EQ(Schedule.size(), 4ul);

  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}
