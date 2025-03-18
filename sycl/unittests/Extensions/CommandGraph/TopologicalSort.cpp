//==------------------------ TopologicalSort.cpp----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

// Checks that the topological sort that is executed at graph finalize
// produces a valid schedule. A schedule is valid if, for every edge (U,V) from
// node A to node B, U comes before V in the ordering.
TEST_F(CommandGraphTest, CheckTopologicalSort) {

  //       Graph structure
  //           (6)-----------------
  //         /    \               |
  //        /      \              |
  //       (3)---->(0)            |
  //        \    /  |  \          |
  //         \  /   |   \         |
  //          (1)  (2)  (5)       |
  //                |             |
  //                |             |
  //               (4) <----------|
  size_t NumNodes = 7;
  experimental::command_graph Graph{Queue.get_context(), Dev};
  auto Node6 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node0 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node4 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.make_edge(Node6, Node3);
  Graph.make_edge(Node6, Node0);
  Graph.make_edge(Node6, Node4);
  Graph.make_edge(Node3, Node0);
  Graph.make_edge(Node3, Node1);
  Graph.make_edge(Node0, Node1);
  Graph.make_edge(Node0, Node2);
  Graph.make_edge(Node0, Node5);
  Graph.make_edge(Node2, Node4);

  // Intentionally make a cycle to make sure that the topological sort at
  // finalize is not affected by cycle checks.
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(Node4, Node6);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  // Make another cycle to check that the cycle checking algorithm is stateless.
  ASSERT_THROW(
      {
        try {
          Graph.make_edge(Node2, Node3);
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  auto ExecGraph = Graph.finalize();

  auto ExecGraphImpl = sycl::detail::getSyclObjImpl(ExecGraph);

  // Creating an executable graph with finalize copies all the nodes to new
  // objects. The new nodes will have an ID equal to OldNodeID + NumNodes. This
  // is implementation dependent but is the only way to test this functionality
  size_t Node6ID = sycl::detail::getSyclObjImpl(Node6)->getID() + NumNodes;
  size_t Node3ID = sycl::detail::getSyclObjImpl(Node3)->getID() + NumNodes;
  size_t Node0ID = sycl::detail::getSyclObjImpl(Node0)->getID() + NumNodes;
  size_t Node1ID = sycl::detail::getSyclObjImpl(Node1)->getID() + NumNodes;
  size_t Node2ID = sycl::detail::getSyclObjImpl(Node2)->getID() + NumNodes;
  size_t Node5ID = sycl::detail::getSyclObjImpl(Node5)->getID() + NumNodes;
  size_t Node4ID = sycl::detail::getSyclObjImpl(Node4)->getID() + NumNodes;

  std::unordered_map<size_t, size_t> mapExecutionIDToTestID = {
      {Node6ID, 6}, {Node3ID, 3}, {Node0ID, 0}, {Node1ID, 1},
      {Node2ID, 2}, {Node5ID, 5}, {Node4ID, 4}};

  // List of all valid topological sorts for this graph.
  std::vector<std::vector<size_t>> ValidSchedules = {
      {6, 3, 0, 1, 2, 4, 5}, {6, 3, 0, 1, 2, 5, 4}, {6, 3, 0, 1, 5, 2, 4},
      {6, 3, 0, 2, 1, 4, 5}, {6, 3, 0, 2, 1, 5, 4}, {6, 3, 0, 2, 4, 1, 5},
      {6, 3, 0, 2, 4, 5, 1}, {6, 3, 0, 2, 5, 1, 4}, {6, 3, 0, 2, 5, 4, 1},
      {6, 3, 0, 5, 1, 2, 4}, {6, 3, 0, 5, 2, 1, 4}, {6, 3, 0, 5, 2, 4, 1}};

  std::vector<size_t> Schedule;
  for (auto &NodeImpl : ExecGraphImpl->getSchedule()) {
    Schedule.push_back(mapExecutionIDToTestID[NodeImpl->getID()]);
  }
  ASSERT_EQ(Schedule.size(), 7ul);

  bool FoundMatchingSchedule = false;
  for (auto &AcceptableSchedule : ValidSchedules) {

    auto Out = std::mismatch(Schedule.begin(), Schedule.end(),
                             AcceptableSchedule.begin());
    if (Out.first == Schedule.end() && Out.second == AcceptableSchedule.end()) {
      FoundMatchingSchedule = true;
      break;
    }
  }

  ASSERT_TRUE(FoundMatchingSchedule);
}
