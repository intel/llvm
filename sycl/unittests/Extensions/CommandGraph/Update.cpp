//==----------------------------- Update.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, UpdatableException) {
  auto Node = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto ExecGraphUpdatable =
      Graph.finalize(experimental::property::graph::updatable{});

  EXPECT_NO_THROW(ExecGraphUpdatable.update(Node));

  auto ExecGraphNoUpdatable = Graph.finalize();

  // Graph without the property should throw
  EXPECT_ANY_THROW(ExecGraphNoUpdatable.update(Node));
}

TEST_F(CommandGraphTest, DynamicParamRegister) {
  // Check that registering a dynamic param with a node from a graph that was
  // not passed to its constructor throws.
  experimental::dynamic_parameter DynamicParam(Graph, int{});

  auto OtherGraph =
      experimental::command_graph(Queue.get_context(), Queue.get_device());
  auto Node = OtherGraph.add([&](sycl::handler &cgh) {
    // This should throw since OtherGraph is not associated with DynamicParam
    EXPECT_ANY_THROW(cgh.set_arg(0, DynamicParam));
    cgh.single_task<TestKernel<>>([]() {});
  });
}

TEST_F(CommandGraphTest, UpdateNodeNotInGraph) {
  // Check that updating a graph with a node which is not part of that graph is
  // an error.

  auto OtherGraph =
      experimental::command_graph(Queue.get_context(), Queue.get_device());
  auto OtherNode = OtherGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto ExecGraph = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_ANY_THROW(ExecGraph.update(OtherNode));
}

TEST_F(CommandGraphTest, UpdateWithUnchangedNode) {
  // Tests that updating a graph with a node with unchanged
  // parameters is not an error

  auto Node = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto ExecGraph = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_NO_THROW(ExecGraph.update(Node));
}

TEST_F(CommandGraphTest, UpdateNodeTypeExceptions) {
  // Check that registering a dynamic parameter with various node types either
  // throws or does not throw as appropriate

  // Allocate some pointers for memory nodes
  int *PtrA = malloc_device<int>(16, Queue);
  int *PtrB = malloc_device<int>(16, Queue);

  experimental::dynamic_parameter DynamicParam{Graph, int{}};

  ASSERT_NO_THROW(auto NodeKernel = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.single_task<TestKernel<>>([]() {});
  }));

  ASSERT_ANY_THROW(auto NodeMemcpy = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.memcpy(PtrA, PtrB, 16 * sizeof(int));
  }));

  ASSERT_ANY_THROW(auto NodeMemset = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.memset(PtrB, 7, 16 * sizeof(int));
  }));

  ASSERT_ANY_THROW(auto NodeMemfill = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.fill(PtrB, 7, 16);
  }));

  ASSERT_ANY_THROW(auto NodePrefetch = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.prefetch(PtrA, 16 * sizeof(int));
  }));

  ASSERT_ANY_THROW(auto NodeMemadvise = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.mem_advise(PtrA, 16 * sizeof(int), 1);
  }));

  ASSERT_ANY_THROW(auto NodeHostTask = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.host_task([]() {});
  }));

  auto NodeEmpty = Graph.add();

  experimental::command_graph Subgraph(Queue.get_context(), Dev);
  // Add an empty node to the subgraph
  Subgraph.add();

  auto SubgraphExec = Subgraph.finalize();
  ASSERT_ANY_THROW(auto NodeSubgraph = Graph.add([&](sycl::handler &cgh) {
    cgh.set_arg(0, DynamicParam);
    cgh.ext_oneapi_graph(SubgraphExec);
  }));
}

TEST_F(CommandGraphTest, UpdateRangeErrors) {
  // Test that the correct errors are throw when trying to update node ranges

  nd_range<1> NDRange{range{128}, range{32}};
  range<1> Range{128};
  auto NodeNDRange = Graph.add([&](sycl::handler &cgh) {
    cgh.parallel_for<TestKernel<>>(NDRange, [](item<1>) {});
  });

  // OK
  EXPECT_NO_THROW(NodeNDRange.update_nd_range(NDRange));
  // Can't update an nd_range node with a range
  EXPECT_ANY_THROW(NodeNDRange.update_range(Range));
  // Can't update with a different number of dimensions
  EXPECT_ANY_THROW(NodeNDRange.update_nd_range(
      nd_range<2>{range<2>{128, 128}, range<2>{32, 32}}));

  auto NodeRange = Graph.add([&](sycl::handler &cgh) {
    cgh.parallel_for<TestKernel<>>(range<1>{128}, [](item<1>) {});
  });

  // OK
  EXPECT_NO_THROW(NodeRange.update_range(Range));
  // Can't update a range node with an nd_range
  EXPECT_ANY_THROW(NodeRange.update_nd_range(NDRange));
  // Can't update with a different number of dimensions
  EXPECT_ANY_THROW(NodeRange.update_range(range<2>{128, 128}));
}
