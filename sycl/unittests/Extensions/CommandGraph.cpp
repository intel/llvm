//==--------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/ext/oneapi/experimental/graph.hpp"
#include <sycl/sycl.hpp>

#include "detail/graph_impl.hpp"

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <gtest/gtest.h>

using namespace sycl;
using namespace sycl::ext::oneapi;

class CommandGraphTest : public ::testing::Test {
public:
  CommandGraphTest()
      : Mock{}, Plat{Mock.getPlatform()}, Dev{Plat.get_devices()[0]},
        Queue{Dev}, Graph{Queue.get_context(), Dev} {}

protected:
  void SetUp() override {}

protected:
  unittest::PiMock Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};

TEST_F(CommandGraphTest, AddNode) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl->MRoots.empty());

  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node1), nullptr);
  ASSERT_FALSE(sycl::detail::getSyclObjImpl(Node1)->isEmpty());
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);
  ASSERT_EQ(*GraphImpl->MRoots.begin(), sycl::detail::getSyclObjImpl(Node1));
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
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front(),
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
    cgh.single_task<class TestKernel1>([=]() { A[0] = 1; });
  });

  // Add independent node
  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  // Add a node that depends on Node1 due to the accessor
  auto Node3 = Graph.add([&](sycl::handler &cgh) {
    sycl::accessor A(Buf, cgh, sycl::write_only, sycl::no_init);
    cgh.single_task<class TestKernel2>([=]() { A[0] = 3; });
  });

  // Guarantee order of independent nodes 1 and 2
  Graph.make_edge(Node2, Node1);

  auto GraphExec = Graph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);

  // The final schedule should contain three nodes in order: 2->1->3
  auto Schedule = GraphExecImpl->getSchedule();
  ASSERT_EQ(Schedule.size(), 3ul);
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node2));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3));
  ASSERT_EQ(Queue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, MakeEdge) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Add two independent nodes
  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
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
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front(),
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
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); },
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
TEST_F(CommandGraphTest, SubGraph) {
  // Add sub-graph with two nodes
  auto Node1Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
  auto Node2Graph = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); },
      {experimental::property::node::depends_on(Node1Graph)});
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  auto Node1MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
  auto Node2MainGraph =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); },
                    {experimental::property::node::depends_on(Node1MainGraph)});
  auto Node3MainGraph = MainGraph.add(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); },
      {experimental::property::node::depends_on(Node2MainGraph)});

  // Assert order of the added sub-graph
  ASSERT_NE(sycl::detail::getSyclObjImpl(Node2MainGraph), nullptr);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2MainGraph)->isEmpty());
  ASSERT_EQ(sycl::detail::getSyclObjImpl(MainGraph)->MRoots.size(), 1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.size(),
            1lu);
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Node1MainGraph)->MSuccessors.front(),
            sycl::detail::getSyclObjImpl(Node1Graph));
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
  ASSERT_EQ(Schedule.size(), 4ul);
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1MainGraph));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node1Graph));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node2Graph));
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, RecordSubGraph) {
  // Record sub-graph with two nodes
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
  auto Node2Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node1Graph);
    cgh.single_task<class TestKernel>([]() {});
  });
  Graph.end_recording(Queue);
  auto GraphExec = Graph.finalize();

  // Add node to main graph followed by sub-graph and another node
  experimental::command_graph MainGraph(Queue.get_context(), Dev);
  MainGraph.begin_recording(Queue);
  auto Node1MainGraph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });
  auto Node2MainGraph = Queue.submit([&](handler &cgh) {
    cgh.depends_on(Node1MainGraph);
    cgh.ext_oneapi_graph(GraphExec);
  });
  auto Node3MainGraph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node2MainGraph);
    cgh.single_task<class TestKernel>([]() {});
  });
  MainGraph.end_recording(Queue);

  // Finalize main graph and check schedule
  auto MainGraphExec = MainGraph.finalize();
  auto MainGraphExecImpl = sycl::detail::getSyclObjImpl(MainGraphExec);
  auto Schedule = MainGraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 4ul);

  // The first and fourth nodes should have events associated with MainGraph but
  // not graph. The second and third nodes were added as a sub-graph and should
  // have events associated with Graph but not MainGraph.
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt),
      sycl::detail::getSyclObjImpl(Node1MainGraph));

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt),
            sycl::detail::getSyclObjImpl(Node1Graph));

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt),
            sycl::detail::getSyclObjImpl(Node2Graph));

  ScheduleIt++;
  ASSERT_ANY_THROW(
      sycl::detail::getSyclObjImpl(Graph)->getEventForNode(*ScheduleIt));
  ASSERT_EQ(
      sycl::detail::getSyclObjImpl(MainGraph)->getEventForNode(*ScheduleIt),
      sycl::detail::getSyclObjImpl(Node3MainGraph));
  ASSERT_EQ(Queue.get_context(), MainGraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmpty) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with a regular node then empty node then a regular
  // node
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 2ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyFirst) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with an empty node then two regular nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 2ul);
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode3);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyLast) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with two regular nodes then an empty node
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class TestKernel>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 2ul);
  ASSERT_EQ(*ScheduleIt, PtrNode1);
  ScheduleIt++;
  ASSERT_EQ(*ScheduleIt, PtrNode2);
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}
