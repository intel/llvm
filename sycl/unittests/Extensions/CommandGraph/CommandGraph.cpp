//==----------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

#include <map>

using namespace sycl;
using namespace sycl::ext::oneapi;

// Helper function for testing weak_object and owner_less for different graph
// types
template <typename T>
void TestGraphTypeInMaps(const T &Graph1, const T &Graph2) {
  weak_object<T> WeakGraph1 = Graph1;
  weak_object<T> WeakGraph2 = Graph2;

  // Use the graph type directly in a map
  std::map<T, int, owner_less<T>> GraphMap;
  ASSERT_NO_THROW(GraphMap.insert({Graph1, 1}));
  ASSERT_NO_THROW(GraphMap.insert({Graph2, 2}));

  // Use the weak_object graph type in a map
  std::map<weak_object<T>, int, owner_less<T>> WeakGraphMap;
  ASSERT_NO_THROW(WeakGraphMap.insert({WeakGraph1, 1}));
  ASSERT_NO_THROW(WeakGraphMap.insert({WeakGraph2, 2}));
}

// Test creating and using ext::oneapi::weak_object and owner_less for
// command_graph class in a map
TEST_F(CommandGraphTest, OwnerLessGraph) {

  using ModifiableGraphT =
      experimental::command_graph<experimental::graph_state::modifiable>;
  using ExecutableGraphT =
      experimental::command_graph<experimental::graph_state::executable>;
  experimental::command_graph Graph2{Queue.get_context(), Dev};

  // Test the default template parameter command_graph explicitly
  TestGraphTypeInMaps<experimental::command_graph<>>(Graph, Graph2);

  TestGraphTypeInMaps<ModifiableGraphT>(Graph, Graph2);

  auto ExecGraph = Graph.finalize();
  auto ExecGraph2 = Graph2.finalize();
  TestGraphTypeInMaps<ExecutableGraphT>(ExecGraph, ExecGraph2);
}

TEST_F(CommandGraphTest, AddNode) {
  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl.MRoots.empty());

  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  ASSERT_FALSE(getSyclObjImpl(Node1)->isEmpty());
  ASSERT_EQ(GraphImpl.MRoots.size(), 1lu);
  ASSERT_EQ(*GraphImpl.MRoots.begin(), &*getSyclObjImpl(Node1));
  ASSERT_TRUE(getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());

  // Add a node which depends on the first
  auto Node2Deps = experimental::property::node::depends_on(Node1);
  ASSERT_EQ(&*getSyclObjImpl(Node2Deps.get_dependencies().front()),
            &*getSyclObjImpl(Node1));
  auto Node2 = Graph.add([&](sycl::handler &cgh) {}, {Node2Deps});
  ASSERT_TRUE(getSyclObjImpl(Node2)->isEmpty());
  ASSERT_EQ(GraphImpl.MRoots.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.front(),
            &*getSyclObjImpl(Node2));
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);

  // Add a third node which depends on both
  auto Node3 =
      Graph.add([&](sycl::handler &cgh) {},
                {experimental::property::node::depends_on(Node1, Node2)});
  ASSERT_TRUE(getSyclObjImpl(Node3)->isEmpty());
  ASSERT_EQ(GraphImpl.MRoots.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);

  // Add a fourth node without any dependencies on the others
  auto Node4 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_TRUE(getSyclObjImpl(Node4)->isEmpty());
  ASSERT_EQ(GraphImpl.MRoots.size(), 2lu);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(getSyclObjImpl(Node3)->MSuccessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);
  ASSERT_TRUE(getSyclObjImpl(Node4)->MPredecessors.empty());
}

TEST_F(CommandGraphTest, Finalize) {
  sycl::buffer<int> Buf(1);
  auto Node1 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::write_only, sycl::no_init);
    cgh.single_task<TestKernel>([]() {});
  });

  // Add independent node
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  // Add a node that depends on Node1 due to the accessor
  auto Node3 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::read_write);
    cgh.single_task<TestKernel>([]() {});
  });

  // Guarantee order of independent nodes 1 and 2
  Graph.make_edge(Node2, Node1);

  auto GraphExec = Graph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);

  // The final schedule should contain three nodes in order: 2->1->3
  auto Schedule = GraphExecImpl.getSchedule();
  ASSERT_EQ(Schedule.size(), 3ul);
  auto ScheduleIt = Schedule.begin();
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*getSyclObjImpl(Node2)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*getSyclObjImpl(Node1)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*getSyclObjImpl(Node3)));
  ASSERT_EQ(Queue.get_context(), GraphExecImpl.getContext());
}

TEST_F(CommandGraphTest, MakeEdge) {
  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  // Add two independent nodes
  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node2 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_EQ(GraphImpl.MRoots.size(), 2ul);
  ASSERT_TRUE(getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node2)->MPredecessors.empty());

  // Connect nodes and verify order
  Graph.make_edge(Node1, Node2);
  ASSERT_EQ(GraphImpl.MRoots.size(), 1ul);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(getSyclObjImpl(Node1)->MSuccessors.front(),
            &*getSyclObjImpl(Node2));
  ASSERT_TRUE(getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_EQ(getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
}

TEST_F(CommandGraphTest, BeginEndRecording) {
  sycl::queue Queue2{Queue.get_context(), Dev};

  // Test throwing behaviour
  // Check that repeatedly calling begin recording on the same queues is an
  // error
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_ANY_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  ASSERT_ANY_THROW(Graph.begin_recording(Queue2));
  // Check we can repeatedly end recording on the same queues
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  // Vector versions
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_ANY_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));

  experimental::command_graph Graph2(Queue.get_context(), Dev);

  Graph.begin_recording(Queue);
  // Trying to record to a second Graph should throw
  ASSERT_ANY_THROW(Graph2.begin_recording(Queue));
  // Trying to end when it is recording to a different graph should throw
  ASSERT_ANY_THROW(Graph2.end_recording(Queue));
  Graph.end_recording(Queue);
}

TEST_F(CommandGraphTest, GetCGCopy) {
  auto Node1 = Graph.add([&](sycl::handler &cgh) {});
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(Node1)});

  // Get copy of CG of Node2 and check equality
  experimental::detail::node_impl &Node2Imp = *getSyclObjImpl(Node2);
  auto Node2CGCopy = Node2Imp.getCGCopy();
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp.MCGType);
  ASSERT_EQ(Node2CGCopy->getType(), sycl::detail::CGType::Kernel);
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp.MCommandGroup->getType());
  ASSERT_EQ(Node2CGCopy->getAccStorage(),
            Node2Imp.MCommandGroup->getAccStorage());
  ASSERT_EQ(Node2CGCopy->getArgsStorage(),
            Node2Imp.MCommandGroup->getArgsStorage());
  ASSERT_EQ(Node2CGCopy->getEvents(), Node2Imp.MCommandGroup->getEvents());
  ASSERT_EQ(Node2CGCopy->getRequirements(),
            Node2Imp.MCommandGroup->getRequirements());
  ASSERT_EQ(Node2CGCopy->getSharedPtrStorage(),
            Node2Imp.MCommandGroup->getSharedPtrStorage());
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword1) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (E)
  ASSERT_EQ(GraphImpl.MRoots.size(), 3lu);
  experimental::detail::node_impl &EmptyImpl = *getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl.MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl.MSuccessors.size(), 0lu);

  experimental::detail::node_impl &Node1Impl = *getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl.MSuccessors[0], &EmptyImpl);
  experimental::detail::node_impl &Node2Impl = *getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl.MSuccessors[0], &EmptyImpl);
  experimental::detail::node_impl &Node3Impl = *getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl.MSuccessors[0], &EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword2) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(Node3Graph)});

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | (4)
  //     \| /
  //     (E)
  ASSERT_EQ(GraphImpl.MRoots.size(), 3lu);
  experimental::detail::node_impl &EmptyImpl = *getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl.MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl.MSuccessors.size(), 0lu);

  experimental::detail::node_impl &Node1Impl = *getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl.MSuccessors[0], &EmptyImpl);
  experimental::detail::node_impl &Node2Impl = *getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl.MSuccessors[0], &EmptyImpl);
  experimental::detail::node_impl &Node3Impl = *getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl.MSuccessors.size(), 1lu);

  experimental::detail::node_impl &Node4Impl = *getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl.MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node4Impl.MSuccessors[0], &EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword3) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(EmptyNode)});

  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //  |\ |
  //  | (E)
  // (3) |
  //    (4)
  ASSERT_EQ(GraphImpl.MRoots.size(), 2lu);
  experimental::detail::node_impl &EmptyImpl = *getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl.MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl.MSuccessors.size(), 1lu);

  experimental::detail::node_impl &Node1Impl = *getSyclObjImpl(Node1Graph);
  experimental::detail::node_impl &Node2Impl = *getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node1Impl.MSuccessors.size(), 2lu);
  ASSERT_EQ(Node2Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl.MSuccessors[0], &EmptyImpl);

  experimental::detail::node_impl &Node3Impl = *getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl.MPredecessors.size(), 1lu);
  ASSERT_EQ(Node3Impl.MPredecessors[0], &Node1Impl);

  experimental::detail::node_impl &Node4Impl = *getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl.MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl.MPredecessors[0], &EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword4) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto EmptyNode2 =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //   \/
  //  (E1) (3)
  //    \  /
  //    (E2)
  ASSERT_EQ(GraphImpl.MRoots.size(), 3lu);
  experimental::detail::node_impl &EmptyImpl = *getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl.MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl.MSuccessors.size(), 1lu);

  experimental::detail::node_impl &Node1Impl = *getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl.MSuccessors[0], &EmptyImpl);
  experimental::detail::node_impl &Node2Impl = *getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl.MSuccessors[0], &EmptyImpl);

  experimental::detail::node_impl &EmptyImpl2 = *getSyclObjImpl(EmptyNode2);
  experimental::detail::node_impl &Node3Impl = *getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl.MPredecessors.size(), 0lu);
  ASSERT_EQ(Node3Impl.MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl.MSuccessors[0], &EmptyImpl2);

  ASSERT_EQ(EmptyImpl2.MPredecessors.size(), 2lu);
}

TEST_F(CommandGraphTest, GraphPartitionsMerging) {
  // Tests that the parition merging algo works as expected in case of backward
  // dependencies
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(NodeA)});
  auto NodeHT1 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeB)});
  auto NodeC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(NodeHT1)});
  auto NodeD = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(NodeB)});
  auto NodeHT2 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeD)});
  auto NodeE = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});
  auto NodeF = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});

  // Backward dependency
  Graph.make_edge(NodeE, NodeHT1);

  auto GraphExec = Graph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);
  auto PartitionsList = GraphExecImpl.getPartitions();
  ASSERT_EQ(PartitionsList.size(), 5ul);
  ASSERT_FALSE(PartitionsList[0]->MIsHostTask);
  ASSERT_TRUE(PartitionsList[1]->MIsHostTask);
  ASSERT_FALSE(PartitionsList[2]->MIsHostTask);
  ASSERT_TRUE(PartitionsList[3]->MIsHostTask);
  ASSERT_FALSE(PartitionsList[4]->MIsHostTask);
}

TEST_F(CommandGraphTest, GetNodeFromEvent) {
  // Test getting a node from a recorded event and using that as a dependency
  // for an explicit node
  Graph.begin_recording(Queue);
  auto EventKernel = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Graph.end_recording();

  experimental::node NodeKernelA =
      experimental::node::get_node_from_event(EventKernel);

  // Add node as a dependency with the property
  auto NodeKernelB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); },
      experimental::property::node::depends_on(NodeKernelA));

  // Test adding a dependency through make_edge
  auto NodeKernelC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  ASSERT_NO_THROW(Graph.make_edge(NodeKernelA, NodeKernelC));

  auto GraphExec = Graph.finalize();
}

// Test adding fill and memset nodes to a graph
TEST_F(CommandGraphTest, FillMemsetNodes) {
  const int Value = 7;
  // Buffer fill
  buffer<int> Buffer{range<1>{1}};
  Buffer.set_write_back(false);

  {
    ext::oneapi::experimental::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {experimental::property::graph::assume_buffer_outlives_graph{}}};

    auto NodeA = Graph.add([&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Value);
    });
    auto NodeB = Graph.add([&](handler &CGH) {
      auto Acc = Buffer.get_access(CGH);
      CGH.fill(Acc, Value);
    });

    experimental::detail::node_impl &NodeAImpl = *getSyclObjImpl(NodeA);
    experimental::detail::node_impl &NodeBImpl = *getSyclObjImpl(NodeB);

    EXPECT_NE(&NodeAImpl, &NodeBImpl);
  }

  // USM
  {
    int *USMPtr = malloc_device<int>(1, Queue);

    // We need to create some differences between nodes because unlike buffer
    // fills they are not differentiated on accessor ptr value.
    auto FillNodeA =
        Graph.add([&](handler &CGH) { CGH.fill(USMPtr, Value, 1); });
    auto FillNodeB =
        Graph.add([&](handler &CGH) { CGH.fill(USMPtr, Value + 1, 1); });
    auto MemsetNodeA =
        Graph.add([&](handler &CGH) { CGH.memset(USMPtr, Value, 1); });
    auto MemsetNodeB =
        Graph.add([&](handler &CGH) { CGH.memset(USMPtr, Value, 2); });

    experimental::detail::node_impl &FillNodeAImpl = *getSyclObjImpl(FillNodeA);
    experimental::detail::node_impl &FillNodeBImpl = *getSyclObjImpl(FillNodeB);
    experimental::detail::node_impl &MemsetNodeAImpl =
        *getSyclObjImpl(MemsetNodeA);
    experimental::detail::node_impl &MemsetNodeBImpl =
        *getSyclObjImpl(MemsetNodeB);

    EXPECT_NE(&FillNodeAImpl, &FillNodeBImpl);
    EXPECT_NE(&MemsetNodeAImpl, &MemsetNodeBImpl);
    sycl::free(USMPtr, Queue);
  }
}

// Test that the expected dependencies are created when recording a graph node
// containing an accessor with mode FirstMode, followed by one containing an
// accessor with mode SecondMode
template <sycl::access_mode FirstMode, sycl::access_mode SecondMode,
          bool ShouldCreateDep>
void testAccessorModeCombo(sycl::queue Queue) {
  buffer<int> Buffer{range<1>{16}};

  ext::oneapi::experimental::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {experimental::property::graph::assume_buffer_outlives_graph{}}};

  Graph.begin_recording(Queue);
  // Create the first node with a write mode
  auto EventFirst = Queue.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<FirstMode>(CGH);
    CGH.single_task<TestKernel>([]() {});
  });

  auto EventSecond = Queue.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<SecondMode>(CGH);
    CGH.single_task<TestKernel>([]() {});
  });
  Graph.end_recording(Queue);

  EXPECT_EQ(Graph.get_root_nodes().size(), ShouldCreateDep ? 1ul : 2ul);

  experimental::node NodeFirst =
      experimental::node::get_node_from_event(EventFirst);
  EXPECT_EQ(NodeFirst.get_predecessors().size(), 0ul);
  EXPECT_EQ(NodeFirst.get_successors().size(), ShouldCreateDep ? 1ul : 0ul);

  experimental::node NodeSecond =
      experimental::node::get_node_from_event(EventSecond);
  EXPECT_EQ(NodeSecond.get_predecessors().size(), ShouldCreateDep ? 1ul : 0ul);
  EXPECT_EQ(NodeSecond.get_successors().size(), 0ul);
}

// Tests that access modes are correctly respected when recording graph nodes
TEST_F(CommandGraphTest, AccessorModeEdges) {

  // Testing access_mode::write and others
  testAccessorModeCombo<access_mode::write, access_mode::discard_read_write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::write, access_mode::discard_write, false>(
      Queue);
  testAccessorModeCombo<access_mode::write, access_mode::read, true>(Queue);
  testAccessorModeCombo<access_mode::write, access_mode::write, false>(Queue);
  testAccessorModeCombo<access_mode::write, access_mode::read_write, true>(
      Queue);
  testAccessorModeCombo<access_mode::write, access_mode::atomic, true>(Queue);

  // Testing access_mode::read and others
  testAccessorModeCombo<access_mode::read, access_mode::discard_read_write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::read, access_mode::discard_write, false>(
      Queue);
  testAccessorModeCombo<access_mode::read, access_mode::read, false>(Queue);
  testAccessorModeCombo<access_mode::read, access_mode::write, false>(Queue);
  testAccessorModeCombo<access_mode::read, access_mode::read_write, false>(
      Queue);
  testAccessorModeCombo<access_mode::read, access_mode::atomic, false>(Queue);

  // Testing access_mode::read_write and others
  testAccessorModeCombo<access_mode::read_write,
                        access_mode::discard_read_write, false>(Queue);
  testAccessorModeCombo<access_mode::read_write, access_mode::discard_write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::read_write, access_mode::read, true>(
      Queue);
  testAccessorModeCombo<access_mode::read_write, access_mode::write, false>(
      Queue);
  testAccessorModeCombo<access_mode::read_write, access_mode::read_write, true>(
      Queue);
  testAccessorModeCombo<access_mode::read_write, access_mode::atomic, true>(
      Queue);

  // Testing access_mode::discard_read_write and others
  testAccessorModeCombo<access_mode::discard_read_write,
                        access_mode::discard_read_write, false>(Queue);
  testAccessorModeCombo<access_mode::discard_read_write,
                        access_mode::discard_write, false>(Queue);
  testAccessorModeCombo<access_mode::discard_read_write, access_mode::read,
                        true>(Queue);
  testAccessorModeCombo<access_mode::discard_read_write, access_mode::write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::discard_read_write,
                        access_mode::read_write, true>(Queue);
  testAccessorModeCombo<access_mode::discard_read_write, access_mode::atomic,
                        true>(Queue);

  // Testing access_mode::discard_write and others
  testAccessorModeCombo<access_mode::discard_write, access_mode::discard_write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::discard_write, access_mode::discard_write,
                        false>(Queue);
  testAccessorModeCombo<access_mode::discard_write, access_mode::read, true>(
      Queue);
  testAccessorModeCombo<access_mode::discard_write, access_mode::write, false>(
      Queue);
  testAccessorModeCombo<access_mode::discard_write, access_mode::read_write,
                        true>(Queue);
  testAccessorModeCombo<access_mode::discard_write, access_mode::atomic, true>(
      Queue);

  // Testing access_mode::atomic and others
  testAccessorModeCombo<access_mode::atomic, access_mode::discard_write, false>(
      Queue);
  testAccessorModeCombo<access_mode::atomic, access_mode::discard_write, false>(
      Queue);
  testAccessorModeCombo<access_mode::atomic, access_mode::read, true>(Queue);
  testAccessorModeCombo<access_mode::atomic, access_mode::write, false>(Queue);
  testAccessorModeCombo<access_mode::atomic, access_mode::read_write, true>(
      Queue);
  testAccessorModeCombo<access_mode::atomic, access_mode::atomic, true>(Queue);
}

// Tests the transitive queue recording behaviour with queue shortcuts.
TEST_F(CommandGraphTest, TransitiveRecordingShortcuts) {
  device Dev;
  context Ctx{{Dev}};
  queue Q1{Ctx, Dev};
  queue Q2{Ctx, Dev};
  queue Q3{Ctx, Dev};

  ext::oneapi::experimental::command_graph Graph1{Q1.get_context(),
                                                  Q1.get_device()};

  Graph1.begin_recording(Q1);

  auto GraphEvent1 = Q1.single_task<class Kernel1>([=] {});
  ASSERT_EQ(Q1.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);
  ASSERT_EQ(Q2.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);
  ASSERT_EQ(Q3.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);

  auto GraphEvent2 = Q2.single_task<class Kernel2>(GraphEvent1, [=] {});
  ASSERT_EQ(Q1.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);
  ASSERT_EQ(Q2.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);
  ASSERT_EQ(Q3.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);

  auto GraphEvent3 = Q3.parallel_for<class Kernel3>(range<1>{1024}, GraphEvent1,
                                                    [=](item<1> Id) {});
  ASSERT_EQ(Q1.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);
  ASSERT_EQ(Q2.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);
  ASSERT_EQ(Q3.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::recording);

  Graph1.end_recording();
  ASSERT_EQ(Q1.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);
  ASSERT_EQ(Q2.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);
  ASSERT_EQ(Q3.ext_oneapi_get_state(),
            ext::oneapi::experimental::queue_state::executing);
}

// Tests that dynamic_work_group_memory.get() will throw on the host side.
TEST_F(CommandGraphTest, DynamicWorkGroupMemoryGet) {
  device Dev;
  context Ctx{{Dev}};
  queue Queue{Ctx, Dev};
  constexpr int LocalSize{32};

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  ext::oneapi::experimental::dynamic_work_group_memory<int[]> DynLocalMem{
      Graph, LocalSize};
  ASSERT_ANY_THROW(DynLocalMem.get());
}

// Tests that dynamic_local_accessor.get() will throw on the host side.
TEST_F(CommandGraphTest, DynamicLocalAccessorGet) {
  device Dev;
  context Ctx{{Dev}};
  queue Queue{Ctx, Dev};
  constexpr int LocalSize{32};

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};

  ext::oneapi::experimental::dynamic_local_accessor<int, 1> DynLocalMem{
      Graph, LocalSize};
  ASSERT_ANY_THROW(DynLocalMem.get());
}
