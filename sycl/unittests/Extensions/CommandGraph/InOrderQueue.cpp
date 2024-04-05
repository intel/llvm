//==----------------------- InOrderQueue.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, InOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode3));
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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

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
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode3));
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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode3));
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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  auto GraphExecImpl = sycl::detail::getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl->getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl->getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithPreviousHostTask) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  auto EventInitial =
      InOrderQueue.submit([&](handler &CGH) { CGH.host_task([=]() {}); });
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto EventLast = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto WaitList = EventLastImpl->getWaitList();
  // Previous task is a host task. Explicit dependency is needed to enforce the
  // execution order.
  ASSERT_EQ(WaitList.size(), 1lu);
  ASSERT_EQ(WaitList[0], EventInitialImpl);
}

TEST_F(CommandGraphTest, InOrderQueueHostTaskAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  auto EventInitial =
      InOrderQueue.submit([&](handler &CGH) { CGH.host_task([=]() {}); });
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is a host task. Explicit dependency is needed to enforce the
  // execution order.
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not a host task. Explicit dependency is still needed
  // to properly handle blocked tasks (the event will be filtered out before
  // submission to the backend).
  ASSERT_EQ(EventLastWaitList.size(), 1lu);
  ASSERT_EQ(EventLastWaitList[0], EventGraphImpl);
}

TEST_F(CommandGraphTest, InOrderQueueMemsetAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Check if device has usm shared allocation.
  if (!InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations))
    return;
  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial = InOrderQueue.memset(TestData, 1, Size * sizeof(int));
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is a memset. Explicit dependency is needed to enforce the
  // execution order.
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not a host task. In Order queue dependency is managed by
  // the backend for non-host kernels.
  ASSERT_EQ(EventLastWaitList.size(), 0lu);
}

TEST_F(CommandGraphTest, InOrderQueueMemcpyAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Check if device has usm shared allocation.
  if (!InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations))
    return;
  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventInitialImpl = sycl::detail::getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode1 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode2 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front().lock(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front().lock(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto PtrNode3 =
      sycl::detail::getSyclObjImpl(InOrderGraph)
          ->getLastInorderNode(sycl::detail::getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front().lock(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front().lock(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

  auto EventGraphImpl = sycl::detail::getSyclObjImpl(EventGraph);
  auto EventGraphWaitList = EventGraphImpl->getWaitList();
  // Previous task is a memcpy. Explicit dependency is needed to enforce the
  // execution order
  ASSERT_EQ(EventGraphWaitList.size(), 1lu);
  ASSERT_EQ(EventGraphWaitList[0], EventInitialImpl);

  auto EventLast =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));
  auto EventLastImpl = sycl::detail::getSyclObjImpl(EventLast);
  auto EventLastWaitList = EventLastImpl->getWaitList();
  // Previous task is not a host task. In Order queue dependency is managed by
  // the backend for non-host kernels.
  ASSERT_EQ(EventLastWaitList.size(), 0lu);
}
