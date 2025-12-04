//==------------------------ Regressions.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

// Tests in this file are based on specific error reports

// Regression test example based on a reported issue with accessor modes not
// being respected in graphs. The test records 3 kernel nodes which all have
// read only dependencies on the same two buffers, with a write dependency on a
// buffer which is different per kernel. This should result in no edges being
// created between these nodes because the accessor mode combinations do not
// indicate a need for dependencies.
// Originally reported here: https://github.com/intel/llvm/issues/12473
TEST_F(CommandGraphTest, AccessorModeRegression) {
  buffer<int> BufferA{range<1>{16}};
  buffer<int> BufferB{range<1>{16}};
  buffer<int> BufferC{range<1>{16}};
  buffer<int> BufferD{range<1>{16}};
  buffer<int> BufferE{range<1>{16}};

  {
    // Buffers must outlive graph
    experimental::command_graph ScopedGraph{
        Queue.get_context(),
        Dev,
        {experimental::property::graph::assume_buffer_outlives_graph{}}};

    ScopedGraph.begin_recording(Queue);

    auto EventA = Queue.submit([&](handler &CGH) {
      auto AccA = BufferA.get_access<access_mode::read>(CGH);
      auto AccB = BufferB.get_access<access_mode::read>(CGH);
      auto AccC = BufferC.get_access<access_mode::write>(CGH);
      CGH.single_task<TestKernel>([]() {});
    });
    auto EventB = Queue.submit([&](handler &CGH) {
      auto AccA = BufferA.get_access<access_mode::read>(CGH);
      auto AccB = BufferB.get_access<access_mode::read>(CGH);
      auto AccD = BufferD.get_access<access_mode::write>(CGH);
      CGH.single_task<TestKernel>([]() {});
    });
    auto EventC = Queue.submit([&](handler &CGH) {
      auto AccA = BufferA.get_access<access_mode::read>(CGH);
      auto AccB = BufferB.get_access<access_mode::read>(CGH);
      auto AccE = BufferE.get_access<access_mode::write>(CGH);
      CGH.single_task<TestKernel>([]() {});
    });

    ScopedGraph.end_recording(Queue);

    experimental::node NodeA = experimental::node::get_node_from_event(EventA);
    EXPECT_EQ(NodeA.get_predecessors().size(), 0ul);
    EXPECT_EQ(NodeA.get_successors().size(), 0ul);
    experimental::node NodeB = experimental::node::get_node_from_event(EventB);
    EXPECT_EQ(NodeB.get_predecessors().size(), 0ul);
    EXPECT_EQ(NodeB.get_successors().size(), 0ul);
    experimental::node NodeC = experimental::node::get_node_from_event(EventC);
    EXPECT_EQ(NodeC.get_predecessors().size(), 0ul);
    EXPECT_EQ(NodeC.get_successors().size(), 0ul);
  }
}

TEST_F(CommandGraphTest, QueueRecordBarrierMultipleGraph) {
  // Test that using barriers recorded from the same queue to
  // different graphs.

  Graph.begin_recording(Queue);
  auto NodeKernel = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Queue.ext_oneapi_submit_barrier({NodeKernel});
  Graph.end_recording(Queue);

  experimental::command_graph<experimental::graph_state::modifiable> GraphB{
      Queue};
  GraphB.begin_recording(Queue);
  auto NodeKernelB = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Queue.ext_oneapi_submit_barrier({NodeKernelB});
  GraphB.end_recording(Queue);

  experimental::command_graph<experimental::graph_state::modifiable> GraphC{
      Queue};
  GraphC.begin_recording(Queue);
  auto NodeKernelC = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Queue.ext_oneapi_submit_barrier();
  GraphC.end_recording(Queue);
}

// Test that the last recorded queue is preserved after cleanup.
// This is a regression test for a bug where getLastRecordedQueue() would
// return nullptr after the recording queues were cleaned up, because the
// previous implementation (getQueue()) looked in the MRecordingQueues set
// which gets cleared on end_recording(). The fix introduces MLastRecordedQueue
// which persists even after cleanup, allowing the executable graph to retrieve
// the queue that was used for recording.
// Originally reported in commit: 0ddf61e3ccaba45ee0af1d1bac12a83328e4015b
TEST_F(CommandGraphTest, LastRecordedQueueAfterCleanup) {
  // Record some work to the graph
  Graph.begin_recording(Queue);
  Queue.submit([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  // Get the graph implementation to check internal state
  auto GraphImpl = getSyclObjImpl(Graph);

  // getLastRecordedQueue() should return the queue that was used for recording
  // even after end_recording() has cleared the recording queues
  auto LastQueue = GraphImpl->getLastRecordedQueue();
  EXPECT_NE(LastQueue, nullptr);
  EXPECT_EQ(LastQueue, getSyclObjImpl(Queue));

  // Finalize the graph - this uses getLastRecordedQueue() internally
  // to set up the executable graph's queue. Before the fix, this could fail
  // if getLastRecordedQueue() returned nullptr.
  auto GraphExec = Graph.finalize();
  auto ExecGraphImpl = *getSyclObjImpl(GraphExec);

  // The executable graph should have the queue from recording
  auto ExecQueueImpl = GraphImplTest::GetQueueImpl(ExecGraphImpl);
  EXPECT_NE(ExecQueueImpl, nullptr);
  EXPECT_EQ(ExecQueueImpl, getSyclObjImpl(Queue));
}
