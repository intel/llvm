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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto ExecGraphUpdatable =
      Graph.finalize(experimental::property::graph::updatable{});

  EXPECT_NO_THROW(ExecGraphUpdatable.update(Node));

  auto ExecGraphNoUpdatable = Graph.finalize();

  // Graph without the property should throw
  EXPECT_ANY_THROW(ExecGraphNoUpdatable.update(Node));
}

TEST_F(CommandGraphTest, DynamicObjRegister) {
  // Check that registering a dynamic object with a node from a graph that was
  // not passed to its constructor does not throw.

  auto CheckRegisterWrongGraph = [&](auto &DynObj) {
    auto OtherGraph =
        experimental::command_graph(Queue.get_context(), Queue.get_device());
    auto Node = OtherGraph.add([&](sycl::handler &cgh) {
      // This should not throw
      EXPECT_NO_THROW(cgh.set_arg(0, DynObj));
      cgh.single_task<TestKernel>([]() {});
    });
  };

  // TODO: Update test when deprecated constructors that take a graph have been
  // removed.
  experimental::dynamic_parameter DynamicParam{Graph, int{}};
  ASSERT_NO_FATAL_FAILURE(CheckRegisterWrongGraph(DynamicParam));

  experimental::dynamic_work_group_memory<int[]> DynamicWorkGroupMem{Graph, 1};
  ASSERT_NO_FATAL_FAILURE(CheckRegisterWrongGraph(DynamicWorkGroupMem));

  experimental::dynamic_local_accessor<int, 1> DynamicLocalAcc{Graph, 1};
  ASSERT_NO_FATAL_FAILURE(CheckRegisterWrongGraph(DynamicLocalAcc));
}

TEST_F(CommandGraphTest, UpdateNodeNotInGraph) {
  // Check that updating a graph with a node which is not part of that graph is
  // an error.

  auto OtherGraph =
      experimental::command_graph(Queue.get_context(), Queue.get_device());
  auto OtherNode = OtherGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto ExecGraph = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_ANY_THROW(ExecGraph.update(OtherNode));
}

TEST_F(CommandGraphTest, UpdateWithUnchangedNode) {
  // Tests that updating a graph with a node with unchanged
  // parameters is not an error

  auto Node = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto ExecGraph = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_NO_THROW(ExecGraph.update(Node));
}

TEST_F(CommandGraphTest, UpdateNodeTypeExceptions) {
  // Check that registering a dynamic object with various node types either
  // throws or does not throw as appropriate

  auto CheckNodeCompatibility = [&](auto &DynObj) {
    // Allocate some pointers for memory nodes
    int *PtrA = malloc_device<int>(16, Queue);
    int *PtrB = malloc_device<int>(16, Queue);

    ASSERT_NO_THROW(auto NodeKernel = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.single_task<TestKernel>([]() {});
    }));

    ASSERT_ANY_THROW(auto NodeMemcpy = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.memcpy(PtrA, PtrB, 16 * sizeof(int));
    }));

    ASSERT_ANY_THROW(auto NodeMemset = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.memset(PtrB, 7, 16 * sizeof(int));
    }));

    ASSERT_ANY_THROW(auto NodeMemfill = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.fill(PtrB, 7, 16);
    }));

    ASSERT_ANY_THROW(auto NodePrefetch = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.prefetch(PtrA, 16 * sizeof(int));
    }));

    ASSERT_ANY_THROW(auto NodeMemadvise = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.mem_advise(PtrA, 16 * sizeof(int), 1);
    }));

    ASSERT_ANY_THROW(auto NodeHostTask = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.host_task([]() {});
    }));

    ASSERT_ANY_THROW(auto NodeBarreriTask = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.ext_oneapi_barrier();
    }));

    Graph.begin_recording(Queue);
    ASSERT_ANY_THROW(auto NodeBarrierTask = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.ext_oneapi_barrier();
    }));
    Graph.end_recording(Queue);

    auto NodeEmpty = Graph.add();

    experimental::command_graph Subgraph(Queue.get_context(),
                                         Queue.get_device());
    // Add an empty node to the subgraph
    Subgraph.add();

    auto SubgraphExec = Subgraph.finalize();
    ASSERT_ANY_THROW(auto NodeSubgraph = Graph.add([&](sycl::handler &cgh) {
      cgh.set_arg(0, DynObj);
      cgh.ext_oneapi_graph(SubgraphExec);
    }));
  };

  experimental::dynamic_parameter DynamicParam{Graph, int{}};
  ASSERT_NO_FATAL_FAILURE(CheckNodeCompatibility(DynamicParam));

  experimental::dynamic_work_group_memory<int[]> DynamicWorkGroupMem{Graph, 1};
  ASSERT_NO_FATAL_FAILURE(CheckNodeCompatibility(DynamicWorkGroupMem));

  experimental::dynamic_local_accessor<int, 1> DynamicLocalAcc{Graph, 1};
  ASSERT_NO_FATAL_FAILURE(CheckNodeCompatibility(DynamicLocalAcc));
}

TEST_F(CommandGraphTest, UpdateRangeErrors) {
  // Test that the correct errors are throw when trying to update node ranges
  nd_range<1> NDRange{range{128}, range{32}};
  range<1> Range{128};
  auto NodeNDRange = Graph.add([&](sycl::handler &cgh) {
    cgh.parallel_for<TestKernel>(NDRange, [](nd_item<1>) {});
  });

  // OK
  EXPECT_NO_THROW(NodeNDRange.update_nd_range(NDRange));
  // OK to update an nd_range node with a range of the same dimension
  EXPECT_NO_THROW(NodeNDRange.update_range(Range));
  // Can't update with a different number of dimensions
  EXPECT_ANY_THROW(NodeNDRange.update_nd_range(
      nd_range<2>{range<2>{128, 128}, range<2>{32, 32}}));
  EXPECT_ANY_THROW(NodeNDRange.update_range(range<3>{32, 32, 1}));

  auto NodeRange = Graph.add([&](sycl::handler &cgh) {
    cgh.parallel_for<TestKernel>(range<1>{128}, [](item<1>) {});
  });

  // OK
  EXPECT_NO_THROW(NodeRange.update_range(Range));
  // OK to update a range node with an nd_range of the same dimension
  EXPECT_NO_THROW(NodeRange.update_nd_range(NDRange));
  // Can't update with a different number of dimensions
  EXPECT_ANY_THROW(NodeRange.update_range(range<2>{128, 128}));
  EXPECT_ANY_THROW(NodeRange.update_nd_range(
      nd_range<3>{range<3>{8, 8, 8}, range<3>{8, 8, 8}}));
}

class WholeGraphUpdateTest : public CommandGraphTest {

protected:
  static constexpr size_t Size = 1024;

  WholeGraphUpdateTest()
      : UpdateGraph{
            Queue.get_context(),
            Dev,
            {experimental::property::graph::assume_buffer_outlives_graph{}}} {}

  experimental::command_graph<experimental::graph_state::modifiable>
      UpdateGraph;

  std::function<void(::sycl::_V1::handler &)> EmptyKernel = [&](handler &CGH) {
    CGH.parallel_for<TestKernel>(range<1>(Size), [=](item<1> Id) {});
  };
};

TEST_F(WholeGraphUpdateTest, NoUpdates) {
  // Test that using an update graph that has no updates is fine.

  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      EmptyKernel, experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeD = UpdateGraph.add(
      EmptyKernel,
      experimental::property::node::depends_on(UpdateNodeB, UpdateNodeC));

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_NO_THROW(GraphExec.update(UpdateGraph));
}

TEST_F(WholeGraphUpdateTest, MoreNodes) {
  // Test that using an update graph that has extra nodes results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      EmptyKernel, experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeD = UpdateGraph.add(
      EmptyKernel,
      experimental::property::node::depends_on(UpdateNodeB, UpdateNodeC));
  // NodeE is the extra node
  auto UpdateNodeE = UpdateGraph.add(EmptyKernel);

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, LessNodes) {
  // Test that using an update graph that has less nodes results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      EmptyKernel, experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  // NodeD is missing in the update

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, ExtraEdges) {
  // Test that using an update graph with extra nodes results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      EmptyKernel, experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeD = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(
                       UpdateNodeA, UpdateNodeB, UpdateNodeC /* Extra Edge */));

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, MissingEdges) {
  // Test that using an update graph with missing edges results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      EmptyKernel, experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeD = UpdateGraph.add(
      EmptyKernel,
      experimental::property::node::depends_on(/* Missing Edge */ UpdateNodeB));

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, UnsupportedNodeType) {
  // Test that using an update graph that contains unsupported node types
  // results in an error.
  buffer<int> Buffer{range<1>{1}};
  auto NodeA = Graph.add(EmptyKernel);
  auto NodeB =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeC =
      Graph.add(EmptyKernel, experimental::property::node::depends_on(NodeA));
  auto NodeD = Graph.add(
      [&](handler &CGH) {
        auto Acc = Buffer.get_access(CGH);
        CGH.fill(Acc, 1);
      },
      experimental::property::node::depends_on(NodeB, NodeC));

  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);
  auto UpdateNodeB = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeC = UpdateGraph.add(
      EmptyKernel, experimental::property::node::depends_on(UpdateNodeA));
  auto UpdateNodeD = UpdateGraph.add(
      [&](handler &CGH) {
        auto Acc = Buffer.get_access(CGH);
        CGH.fill(Acc, 1);
      },
      experimental::property::node::depends_on(UpdateNodeB, UpdateNodeC));

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, WrongContext) {
  // Test that using an update graph that was created with a different context
  // (when compared to the original graph) results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});

  context OtherContext(Dev);
  experimental::command_graph<experimental::graph_state::modifiable>
      WrongContextGraph{
          OtherContext,
          Dev,
          {experimental::property::graph::assume_buffer_outlives_graph{}}};

  auto UpdateNodeA = WrongContextGraph.add(EmptyKernel);

  EXPECT_THROW(GraphExec.update(WrongContextGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, WrongDevice) {
  // Test that using an update graph that was created with a different device
  // (when compared to the original graph) results in an error.

  auto devices = device::get_devices();
  if (devices.size() > 1) {

    device &OtherDevice = (devices[0] == Dev ? devices[1] : devices[0]);

    auto NodeA = Graph.add(EmptyKernel);
    auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});

    experimental::command_graph<experimental::graph_state::modifiable>
        WrongDeviceGraph{
            Queue.get_context(),
            OtherDevice,
            {experimental::property::graph::assume_buffer_outlives_graph{}}};

    auto UpdateNodeA = WrongDeviceGraph.add(EmptyKernel);

    EXPECT_THROW(GraphExec.update(WrongDeviceGraph), sycl::exception);
  }
}

TEST_F(WholeGraphUpdateTest, MissingUpdatableProperty) {
  // Test that updating a graph that was not created with the updatable property
  // results in an error.

  auto NodeA = Graph.add(EmptyKernel);
  auto UpdateNodeA = UpdateGraph.add(EmptyKernel);

  auto GraphExec = Graph.finalize();
  EXPECT_THROW(GraphExec.update(UpdateGraph), sycl::exception);
}

TEST_F(WholeGraphUpdateTest, EmptyNode) {
  // Test that updating a graph that has an empty node is not an error
  auto NodeEmpty = Graph.add();
  auto UpdateNodeEmpty = UpdateGraph.add();

  auto NodeKernel = Graph.add(EmptyKernel);
  auto UpdateNodeKernel = UpdateGraph.add(EmptyKernel);

  auto GraphExec = Graph.finalize(experimental::property::graph::updatable{});
  GraphExec.update(UpdateGraph);
}

// Vars and callbacks for tracking how many times mocked functions are called
static int GetInfoCount = 0;
static int AppendKernelLaunchCount = 0;
static ur_result_t redefinedCommandBufferGetInfoExpAfter(void *pParams) {
  GetInfoCount++;
  return UR_RESULT_SUCCESS;
}
static ur_result_t
redefinedCommandBufferAppendKernelLaunchExpAfter(void *pParams) {
  AppendKernelLaunchCount++;
  return UR_RESULT_SUCCESS;
}

TEST_F(CommandGraphTest, CheckFinalizeBehavior) {
  // Check that both finalize with and without updatable property work as
  // expected
  auto Node = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  mock::getCallbacks().set_after_callback(
      "urCommandBufferGetInfoExp", &redefinedCommandBufferGetInfoExpAfter);
  mock::getCallbacks().set_after_callback(
      "urCommandBufferAppendKernelLaunchExp",
      &redefinedCommandBufferAppendKernelLaunchExpAfter);

  ASSERT_NO_THROW(Graph.finalize(experimental::property::graph::updatable{}));
  // GetInfo and AppendKernelLaunch should be called once each time a node is
  // added to a command buffer during finalization
  ASSERT_EQ(GetInfoCount, 1);
  ASSERT_EQ(AppendKernelLaunchCount, 1);

  ASSERT_NO_THROW(Graph.finalize());
  ASSERT_EQ(GetInfoCount, 2);
  ASSERT_EQ(AppendKernelLaunchCount, 2);
}
