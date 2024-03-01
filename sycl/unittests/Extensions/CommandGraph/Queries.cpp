//==---------------------------- Queries.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, QueueState) {
  experimental::queue_state State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::executing);

  experimental::command_graph Graph{Queue.get_context(), Queue.get_device()};
  Graph.begin_recording(Queue);
  State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::recording);

  Graph.end_recording();
  State = Queue.ext_oneapi_get_state();
  ASSERT_EQ(State, experimental::queue_state::executing);
}

TEST_F(CommandGraphTest, GetNodeQueries) {
  // Tests graph and node queries for correctness

  // Add some nodes to the graph for testing and test after each addition.
  auto RootA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  {
    auto GraphRoots = Graph.get_root_nodes();
    auto GraphNodes = Graph.get_nodes();
    ASSERT_EQ(GraphRoots.size(), 1lu);
    ASSERT_EQ(GraphNodes.size(), 1lu);
  }
  auto RootB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  {
    auto GraphRoots = Graph.get_root_nodes();
    auto GraphNodes = Graph.get_nodes();
    ASSERT_EQ(GraphRoots.size(), 2lu);
    ASSERT_EQ(GraphNodes.size(), 2lu);
  }
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(RootA, RootB)});
  {
    auto GraphRoots = Graph.get_root_nodes();
    auto GraphNodes = Graph.get_nodes();
    ASSERT_EQ(GraphRoots.size(), 2lu);
    ASSERT_EQ(GraphNodes.size(), 3lu);
  }
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(RootB)});
  {
    auto GraphRoots = Graph.get_root_nodes();
    auto GraphNodes = Graph.get_nodes();
    ASSERT_EQ(GraphRoots.size(), 2lu);
    ASSERT_EQ(GraphNodes.size(), 4lu);
  }
  auto RootC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  {
    auto GraphRoots = Graph.get_root_nodes();
    auto GraphNodes = Graph.get_nodes();
    ASSERT_EQ(GraphRoots.size(), 3lu);
    ASSERT_EQ(GraphNodes.size(), 5lu);
  }

  ASSERT_EQ(RootA.get_predecessors().size(), 0lu);
  ASSERT_EQ(RootA.get_successors().size(), 1lu);
  ASSERT_EQ(RootB.get_predecessors().size(), 0lu);
  ASSERT_EQ(RootB.get_successors().size(), 2lu);
  ASSERT_EQ(RootC.get_predecessors().size(), 0lu);
  ASSERT_EQ(RootC.get_successors().size(), 0lu);
  ASSERT_EQ(NodeA.get_predecessors().size(), 2lu);
  ASSERT_EQ(NodeA.get_successors().size(), 0lu);
  ASSERT_EQ(NodeB.get_predecessors().size(), 1lu);
  ASSERT_EQ(NodeB.get_successors().size(), 0lu);

  // List of nodes that we've added in the order they were added.
  std::vector<experimental::node> NodeList{RootA, RootB, NodeA, NodeB, RootC};
  auto GraphNodes = Graph.get_nodes();

  // Check ordering of all nodes is correct
  for (size_t i = 0; i < GraphNodes.size(); i++) {
    ASSERT_EQ(sycl::detail::getSyclObjImpl(GraphNodes[i]),
              sycl::detail::getSyclObjImpl(NodeList[i]));
  }
}

TEST_F(CommandGraphTest, NodeTypeQueries) {

  // Allocate some pointers for testing memory nodes
  int *PtrA = malloc_device<int>(16, Queue);
  int *PtrB = malloc_device<int>(16, Queue);

  auto NodeKernel = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  ASSERT_EQ(NodeKernel.get_type(), experimental::node_type::kernel);

  auto NodeMemcpy = Graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(PtrA, PtrB, 16 * sizeof(int)); });
  ASSERT_EQ(NodeMemcpy.get_type(), experimental::node_type::memcpy);

  auto NodeMemset = Graph.add(
      [&](sycl::handler &cgh) { cgh.memset(PtrB, 7, 16 * sizeof(int)); });
  ASSERT_EQ(NodeMemset.get_type(), experimental::node_type::memset);

  auto NodeMemfill =
      Graph.add([&](sycl::handler &cgh) { cgh.fill(PtrB, 7, 16); });
  ASSERT_EQ(NodeMemfill.get_type(), experimental::node_type::memfill);

  auto NodePrefetch = Graph.add(
      [&](sycl::handler &cgh) { cgh.prefetch(PtrA, 16 * sizeof(int)); });
  ASSERT_EQ(NodePrefetch.get_type(), experimental::node_type::prefetch);

  auto NodeMemadvise = Graph.add(
      [&](sycl::handler &cgh) { cgh.mem_advise(PtrA, 16 * sizeof(int), 1); });
  ASSERT_EQ(NodeMemadvise.get_type(), experimental::node_type::memadvise);

  // Use queue recording for barrier since it is not supported in explicit API
  Graph.begin_recording(Queue);
  auto EventBarrier =
      Queue.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });
  Graph.end_recording();

  auto NodeBarrier = experimental::node::get_node_from_event(EventBarrier);
  ASSERT_EQ(NodeBarrier.get_type(),
            experimental::node_type::ext_oneapi_barrier);

  auto NodeHostTask =
      Graph.add([&](sycl::handler &cgh) { cgh.host_task([]() {}); });
  ASSERT_EQ(NodeHostTask.get_type(), experimental::node_type::host_task);

  auto NodeEmpty = Graph.add();
  ASSERT_EQ(NodeEmpty.get_type(), experimental::node_type::empty);

  experimental::command_graph Subgraph(Queue.get_context(), Dev);
  // Add an empty node to the subgraph
  Subgraph.add();

  auto SubgraphExec = Subgraph.finalize();
  auto NodeSubgraph = Graph.add(
      [&](sycl::handler &cgh) { cgh.ext_oneapi_graph(SubgraphExec); });
  ASSERT_EQ(NodeSubgraph.get_type(), experimental::node_type::subgraph);
}
