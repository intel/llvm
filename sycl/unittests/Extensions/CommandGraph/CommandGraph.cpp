//==----------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, AddNode) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl->MRoots.empty());

  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node1), nullptr);
  ASSERT_FALSE(sycl::detail::getSyclObjImpl(Node1)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ((*GraphImpl->MRoots.begin()).lock(),
            sycl::detail::getSyclObjImpl(Node1));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());

  // Add a node which depends on the first
  auto Node2Deps = experimental::property::node::depends_on(Node1);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2Deps.get_dependencies().front()),
            sycl::detail::getSyclObjImpl(Node1));
  auto Node2 = Graph.add([&](sycl::handler &cgh) {}, {Node2Deps});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front().lock(),
            sycl::detail::getSyclObjImpl(Node2));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);

  // Add a third node which depends on both
  auto Node3 =
      Graph.add([&](sycl::handler &cgh) {},
                {experimental::property::node::depends_on(Node1, Node2)});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node3), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node3)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);

  // Add a fourth node without any dependencies on the others
  auto Node4 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node4), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node4)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 2lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size(), 1lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node3)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node3)->MPredecessors.size(), 2lu);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node4)->MPredecessors.empty());
}

TEST_F(CommandGraphTest, Finalize) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  sycl::buffer<int> Buf(1);
  auto Node1 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::write_only, sycl::no_init);
    cgh.single_task<TestKernel<>>([]() {});
  });

  // Add independent node
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Add a node that depends on Node1 due to the accessor
  auto Node3 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::read_write);
    cgh.single_task<TestKernel<>>([]() {});
  });

  // Guarantee order of independent nodes 1 and 2
  Graph.make_edge(Node2, Node1);

  auto GraphExec = Graph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);

  // The final schedule should contain three nodes in order: 2->1->3
  auto Schedule = GraphExecImpl->getSchedule();
  ASSERT_EQ(Schedule.size(), 3ul);
  auto ScheduleIt = Schedule.begin();
  ASSERT_TRUE((*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node2)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node1)));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(sycl::detail::getSyclObjImpl(Node3)));
  ASSERT_EQ(Queue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, MakeEdge) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Add two independent nodes
  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_EQ(GraphImpl->MRoots.size(), 2ul);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.empty());

  // Connect nodes and verify order
  Graph.make_edge(Node1, Node2);
  ASSERT_EQ(GraphImpl->MRoots.size(), 1ul);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front().lock(),
            sycl::detail::getSyclObjImpl(Node2));
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MPredecessors.empty());
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.empty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size(), 1lu);
}

TEST_F(CommandGraphTest, BeginEndRecording) {
  sycl::queue Queue2{Queue.get_context(), Dev};

  // Test throwing behaviour
  // Check we can repeatedly begin recording on the same queues
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  // Check we can repeatedly end recording on the same queues
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  // Vector versions
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));

  experimental::command_graph Graph2(Queue.get_context(), Dev);

  Graph.begin_recording(Queue);
  // Trying to record to a second Graph should throw
  ASSERT_ANY_THROW(Graph2.begin_recording(Queue));
  // Trying to end when it is recording to a different graph should throw
  ASSERT_ANY_THROW(Graph2.end_recording(Queue));
  Graph.end_recording(Queue);

  // Testing return values of begin and end recording
  // Queue should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue));

  // Queue2 should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue2));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // Queue should have changed state so should return true
  ASSERT_TRUE(Graph.end_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.end_recording(Queue));

  // Should end recording on Queue2
  ASSERT_TRUE(Graph.end_recording());
  // State should not change on Queue2 now
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // Testing vector begin and end
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.begin_recording(Queue));
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // End recording on both
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.end_recording(Queue));
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // First add one single queue
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // Vector begin should still return true as Queue2 has state changed
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // End recording on Queue2
  ASSERT_TRUE(Graph.end_recording(Queue2));
  // Vector end should still return true as Queue will have state changed
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
}

TEST_F(CommandGraphTest, GetCGCopy) {
  auto Node1 = Graph.add([&](sycl::handler &cgh) {});
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1)});

  // Get copy of CG of Node2 and check equality
  auto Node2Imp = sycl::detail::getSyclObjImpl(Node2);
  auto Node2CGCopy = Node2Imp->getCGCopy();
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp->MCGType);
  ASSERT_EQ(Node2CGCopy->getType(), sycl::detail::CG::Kernel);
  ASSERT_EQ(Node2CGCopy->getType(), Node2Imp->MCommandGroup->getType());
  ASSERT_EQ(Node2CGCopy->getAccStorage(),
            Node2Imp->MCommandGroup->getAccStorage());
  ASSERT_EQ(Node2CGCopy->getArgsStorage(),
            Node2Imp->MCommandGroup->getArgsStorage());
  ASSERT_EQ(Node2CGCopy->getEvents(), Node2Imp->MCommandGroup->getEvents());
  ASSERT_EQ(Node2CGCopy->getRequirements(),
            Node2Imp->MCommandGroup->getRequirements());
  ASSERT_EQ(Node2CGCopy->getSharedPtrStorage(),
            Node2Imp->MCommandGroup->getSharedPtrStorage());
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword1) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (E)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 0lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MSuccessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword2) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node3Graph)});

  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | (4)
  //     \| /
  //     (E)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 3lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 0lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);

  auto Node4Impl = sycl::detail::getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MSuccessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword3) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto Node4Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(EmptyNode)});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //  |\ |
  //  | (E)
  // (3) |
  //    (4)
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 1lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 2lu);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);

  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MPredecessors[0].lock(), Node1Impl);

  auto Node4Impl = sycl::detail::getSyclObjImpl(Node4Graph);
  ASSERT_EQ(Node4Impl->MPredecessors.size(), 1lu);
  ASSERT_EQ(Node4Impl->MPredecessors[0].lock(), EmptyImpl);
}

TEST_F(CommandGraphTest, DependencyLeavesKeyword4) {
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});
  auto Node3Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EmptyNode2 =
      Graph.add([&](sycl::handler &cgh) { /*empty node */ },
                {experimental::property::node::depends_on_all_leaves()});

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1)(2)
  //   \/
  //  (E1) (3)
  //    \  /
  //    (E2)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  auto EmptyImpl = sycl::detail::getSyclObjImpl(EmptyNode);
  ASSERT_EQ(EmptyImpl->MPredecessors.size(), 2lu);
  ASSERT_EQ(EmptyImpl->MSuccessors.size(), 1lu);

  auto Node1Impl = sycl::detail::getSyclObjImpl(Node1Graph);
  ASSERT_EQ(Node1Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node1Impl->MSuccessors[0].lock(), EmptyImpl);
  auto Node2Impl = sycl::detail::getSyclObjImpl(Node2Graph);
  ASSERT_EQ(Node2Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node2Impl->MSuccessors[0].lock(), EmptyImpl);

  auto EmptyImpl2 = sycl::detail::getSyclObjImpl(EmptyNode2);
  auto Node3Impl = sycl::detail::getSyclObjImpl(Node3Graph);
  ASSERT_EQ(Node3Impl->MPredecessors.size(), 0lu);
  ASSERT_EQ(Node3Impl->MSuccessors.size(), 1lu);
  ASSERT_EQ(Node3Impl->MSuccessors[0].lock(), EmptyImpl2);

  ASSERT_EQ(EmptyImpl2->MPredecessors.size(), 2lu);
}

TEST_F(CommandGraphTest, GraphPartitionsMerging) {
  // Tests that the parition merging algo works as expected in case of backward
  // dependencies
  auto NodeA = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeA)});
  auto NodeHT1 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeB)});
  auto NodeC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT1)});
  auto NodeD = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeB)});
  auto NodeHT2 = Graph.add([&](sycl::handler &cgh) { cgh.host_task([=]() {}); },
                           {experimental::property::node::depends_on(NodeD)});
  auto NodeE = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});
  auto NodeF = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(NodeHT2)});

  // Backward dependency
  Graph.make_edge(NodeE, NodeHT1);

  auto GraphExec = Graph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto PartitionsList = GraphExecImpl->getPartitions();
  ASSERT_EQ(PartitionsList.size(), 5ul);
  ASSERT_FALSE(PartitionsList[0]->isHostTask());
  ASSERT_TRUE(PartitionsList[1]->isHostTask());
  ASSERT_FALSE(PartitionsList[2]->isHostTask());
  ASSERT_TRUE(PartitionsList[3]->isHostTask());
  ASSERT_FALSE(PartitionsList[4]->isHostTask());
}

TEST_F(CommandGraphTest, GetNodeFromEvent) {
  // Test getting a node from a recorded event and using that as a dependency
  // for an explicit node
  Graph.begin_recording(Queue);
  auto EventKernel = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  experimental::node NodeKernelA =
      experimental::node::get_node_from_event(EventKernel);

  // Add node as a dependency with the property
  auto NodeKernelB = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      experimental::property::node::depends_on(NodeKernelA));

  // Test adding a dependency through make_edge
  auto NodeKernelC = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
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

    auto NodeAImpl = sycl::detail::getSyclObjImpl(NodeA);
    auto NodeBImpl = sycl::detail::getSyclObjImpl(NodeB);

    // Check Operator==
    EXPECT_EQ(NodeAImpl, NodeAImpl);
    EXPECT_NE(NodeAImpl, NodeBImpl);
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

    auto FillNodeAImpl = sycl::detail::getSyclObjImpl(FillNodeA);
    auto FillNodeBImpl = sycl::detail::getSyclObjImpl(FillNodeB);
    auto MemsetNodeAImpl = sycl::detail::getSyclObjImpl(MemsetNodeA);
    auto MemsetNodeBImpl = sycl::detail::getSyclObjImpl(MemsetNodeB);

    // Check Operator==
    EXPECT_EQ(FillNodeAImpl, FillNodeAImpl);
    EXPECT_EQ(FillNodeBImpl, FillNodeBImpl);
    EXPECT_NE(FillNodeAImpl, FillNodeBImpl);

    EXPECT_EQ(MemsetNodeAImpl, MemsetNodeAImpl);
    EXPECT_EQ(MemsetNodeBImpl, MemsetNodeBImpl);
    EXPECT_NE(MemsetNodeAImpl, MemsetNodeBImpl);
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
    CGH.single_task<TestKernel<>>([]() {});
  });

  auto EventSecond = Queue.submit([&](handler &CGH) {
    auto Acc = Buffer.get_access<SecondMode>(CGH);
    CGH.single_task<TestKernel<>>([]() {});
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
