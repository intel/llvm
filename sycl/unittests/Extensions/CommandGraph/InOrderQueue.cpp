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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl.getSchedule();
  auto ScheduleIt = Schedule.begin();
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode3));
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl.getContext());
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
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl.getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode3));
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl.getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyFirst) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with an empty node then two regular nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl.getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode3));
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl.getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithEmptyLast) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with two regular nodes then an empty node
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit([&](sycl::handler &cgh) {});

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  // Note that empty nodes are not scheduled
  auto GraphExec = InOrderGraph.finalize();
  experimental::detail::exec_graph_impl &GraphExecImpl =
      *getSyclObjImpl(GraphExec);
  auto Schedule = GraphExecImpl.getSchedule();
  auto ScheduleIt = Schedule.begin();
  // the schedule list contains all types of nodes (even empty nodes)
  ASSERT_EQ(Schedule.size(), 3ul);
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode1));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isSimilar(*PtrNode2));
  ScheduleIt++;
  ASSERT_TRUE((*ScheduleIt)->isEmpty());
  ASSERT_EQ(InOrderQueue.get_context(), GraphExecImpl.getContext());
}

TEST_F(CommandGraphTest, InOrderQueueWithPreviousHostTask) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Event dependency build depends on host task completion. Making it
  // predictable with mutex in host task.
  std::mutex HostTaskMutex;
  std::unique_lock<std::mutex> Lock(HostTaskMutex, std::defer_lock);
  Lock.lock();
  auto EventInitial = InOrderQueue.submit([&](handler &CGH) {
    CGH.host_task([&HostTaskMutex]() {
      std::lock_guard<std::mutex> HostTaskLock(HostTaskMutex);
    });
  });
  sycl::detail::event_impl &EventInitialImpl = *getSyclObjImpl(EventInitial);

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto EventLast = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  sycl::detail::event_impl &EventLastImpl = *getSyclObjImpl(EventLast);
  auto WaitList = EventLastImpl.getWaitList();
  Lock.unlock();
  // Previous task is a host task. Explicit dependency is needed to enforce the
  // execution order.
  ASSERT_EQ(WaitList.size(), 1lu);
  ASSERT_EQ(WaitList[0].get(), &EventInitialImpl);
  InOrderQueue.wait();
}

TEST_F(CommandGraphTest, InOrderQueueHostTaskAndGraph) {
  auto TestBody = [&](bool BlockHostTask) {
    sycl::property_list Properties{sycl::property::queue::in_order()};
    sycl::queue InOrderQueue{Dev, Properties};
    experimental::command_graph<experimental::graph_state::modifiable>
        InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};
    // Event dependency build depends on host task completion. Making it
    // predictable with mutex in host task.
    std::mutex HostTaskMutex;
    std::unique_lock<std::mutex> Lock(HostTaskMutex, std::defer_lock);
    if (BlockHostTask)
      Lock.lock();
    auto EventInitial = InOrderQueue.submit([&](handler &CGH) {
      CGH.host_task([&HostTaskMutex]() {
        std::lock_guard<std::mutex> HostTaskLock(HostTaskMutex);
      });
    });

    // Record in-order queue with three nodes.
    InOrderGraph.begin_recording(InOrderQueue);
    auto Node1Graph = InOrderQueue.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

    auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                        ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
    ASSERT_NE(PtrNode1, nullptr);
    ASSERT_TRUE(PtrNode1->MPredecessors.empty());

    auto Node2Graph = InOrderQueue.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

    auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                        ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
    ASSERT_NE(PtrNode2, nullptr);
    ASSERT_NE(PtrNode2, PtrNode1);
    ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
    ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
    ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
    ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

    auto Node3Graph = InOrderQueue.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

    auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                        ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
    ASSERT_NE(PtrNode3, nullptr);
    ASSERT_NE(PtrNode3, PtrNode2);
    ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
    ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
    ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
    ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

    InOrderGraph.end_recording(InOrderQueue);

    auto InOrderGraphExec = InOrderGraph.finalize();

    if (!BlockHostTask)
      EventInitial.wait();
    auto EventGraph = InOrderQueue.submit(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });

    auto EventLast = InOrderQueue.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
    sycl::detail::event_impl &EventLastImpl = *getSyclObjImpl(EventLast);
    auto EventLastWaitList = EventLastImpl.getWaitList();
    // Previous task is not a host task. Explicit dependency is still needed
    // to properly handle blocked tasks (the event will be filtered out before
    // submission to the backend).
    if (BlockHostTask)
      Lock.unlock();
    EventLast.wait();
  };

  TestBody(false);
  TestBody(true);
}

TEST_F(CommandGraphTest, InOrderQueueMemsetAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // The mock adapter should return true for shared USM allocation support by
  // default. If this fails it means this test needs to redefine the device info
  // query.
  ASSERT_TRUE(
      InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations));

  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial = InOrderQueue.memset(TestData, 1, Size * sizeof(int));

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });
}

TEST_F(CommandGraphTest, InOrderQueueMemcpyAndGraph) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // The mock adapter should return true for shared USM allocation support by
  // default. If this fails it means this test needs to redefine the device info
  // query.
  ASSERT_TRUE(
      InOrderQueue.get_device().has(sycl::aspect::usm_shared_allocations));

  size_t Size = 128;
  std::vector<int> TestDataHost(Size);
  int *TestData = sycl::malloc_shared<int>(Size, InOrderQueue);

  auto EventInitial =
      InOrderQueue.memcpy(TestData, TestDataHost.data(), Size * sizeof(int));

  // Record in-order queue with three nodes.
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode1 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode1, nullptr);
  ASSERT_TRUE(PtrNode1->MPredecessors.empty());

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode2 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode2, nullptr);
  ASSERT_NE(PtrNode2, PtrNode1);
  ASSERT_EQ(PtrNode1->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode1->MSuccessors.front(), PtrNode2);
  ASSERT_EQ(PtrNode2->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MPredecessors.front(), PtrNode1);

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto PtrNode3 = getSyclObjImpl(InOrderGraph)
                      ->getLastInorderNode(&*getSyclObjImpl(InOrderQueue));
  ASSERT_NE(PtrNode3, nullptr);
  ASSERT_NE(PtrNode3, PtrNode2);
  ASSERT_EQ(PtrNode2->MSuccessors.size(), 1lu);
  ASSERT_EQ(PtrNode2->MSuccessors.front(), PtrNode3);
  ASSERT_EQ(PtrNode3->MPredecessors.size(), 1lu);
  ASSERT_EQ(PtrNode3->MPredecessors.front(), PtrNode2);

  InOrderGraph.end_recording(InOrderQueue);

  auto InOrderGraphExec = InOrderGraph.finalize();
  auto EventGraph = InOrderQueue.submit(
      [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(InOrderGraphExec); });
}

// Validate that enqueuing a graph with
// sycl::ext::oneapi::experimental::execute_graph using an in-order queue,
// does not request a signaling event from the UR backend and has no event
// dependencies.
TEST_F(CommandGraphTest, InOrderQueueEventless) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();

  auto beforeUrEnqueueCommandBufferExp = [](void *pParams) -> ur_result_t {
    auto params =
        *static_cast<ur_enqueue_command_buffer_exp_params_t *>(pParams);
    EXPECT_TRUE(*params.pnumEventsInWaitList == 0);
    EXPECT_TRUE(*params.pphEventWaitList == nullptr);
    EXPECT_TRUE(*params.pphEvent == nullptr);

    return UR_RESULT_SUCCESS;
  };

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueCommandBufferExp",
                                           beforeUrEnqueueCommandBufferExp);

  const size_t Iterations = 5;
  for (size_t I = 0; I < Iterations; ++I) {
    sycl::ext::oneapi::experimental::execute_graph(InOrderQueue, GraphExec);
  }
}

// Validate that if an event is requested when enqueueing a graph with
// sycl::ext::oneapi::experimental::submit_with_event with an in-order queue,
// the implementation requests a signal event but doesn't wait on any events
// dependencies.
TEST_F(CommandGraphTest, InOrderQueueRequestEvent) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  InOrderGraph.end_recording(InOrderQueue);

  // Finalize main graph and check schedule
  auto GraphExec = InOrderGraph.finalize();

  auto beforeUrEnqueueCommandBufferExp = [](void *pParams) -> ur_result_t {
    auto params =
        *static_cast<ur_enqueue_command_buffer_exp_params_t *>(pParams);
    EXPECT_TRUE(*params.pnumEventsInWaitList == 0);
    EXPECT_TRUE(*params.pphEventWaitList == nullptr);
    EXPECT_TRUE(*params.pphEvent != nullptr);

    return UR_RESULT_SUCCESS;
  };

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueCommandBufferExp",
                                           beforeUrEnqueueCommandBufferExp);

  const size_t Iterations = 5;
  std::vector<sycl::event> OutputEvents;

  for (size_t I = 0; I < Iterations; ++I) {
    OutputEvents.push_back(sycl::ext::oneapi::experimental::submit_with_event(
        InOrderQueue,
        [&](sycl::handler &cgh) { cgh.ext_oneapi_graph(GraphExec); }));
  }
}

// Validate that enqueuing a graph using an in-order queue with an event
// dependency does not request a signaling event from the UR backend and has
// 1 event dependency.
TEST_F(CommandGraphTest, InOrderQueueEventlessWithDependency) {
  device Dev{};
  context Context{Dev};

  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Context, Dev, Properties};
  sycl::queue OtherQueue{Context, Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  // Record in-order queue with three nodes
  InOrderGraph.begin_recording(InOrderQueue);
  auto Node1Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node2Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  auto Node3Graph = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  InOrderGraph.end_recording(InOrderQueue);

  auto GraphExec = InOrderGraph.finalize();

  auto beforeUrEnqueueCommandBufferExp = [](void *pParams) -> ur_result_t {
    auto params =
        *static_cast<ur_enqueue_command_buffer_exp_params_t *>(pParams);
    EXPECT_TRUE(*params.pnumEventsInWaitList == 1);
    EXPECT_TRUE(*params.pphEvent == nullptr);

    return UR_RESULT_SUCCESS;
  };

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueCommandBufferExp",
                                           beforeUrEnqueueCommandBufferExp);

  sycl::event Event = sycl::ext::oneapi::experimental::submit_with_event(
      OtherQueue, [&](sycl::handler &CGH) {
        sycl::ext::oneapi::experimental::single_task<class TestKernel>(
            CGH, [=]() {});
      });

  const size_t Iterations = 5;
  for (size_t I = 0; I < Iterations; ++I) {
    sycl::ext::oneapi::experimental::submit(
        InOrderQueue, [&](sycl::handler &CGH) {
          CGH.depends_on(Event);
          sycl::ext::oneapi::experimental::execute_graph(CGH, GraphExec);
        });
  }
}
