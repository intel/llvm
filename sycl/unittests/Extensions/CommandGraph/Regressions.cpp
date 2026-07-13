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
TEST_F(CommandGraphTest, LastRecordedQueueAfterCleanup) {
  // Record some work to the graph
  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
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
  experimental::detail::exec_graph_impl &ExecGraphImpl =
      *getSyclObjImpl(GraphExec);

  // The executable graph should have the queue from recording
  auto ExecQueueImpl = GraphImplTest::GetQueueImpl(ExecGraphImpl);
  EXPECT_NE(ExecQueueImpl, nullptr);
  EXPECT_EQ(ExecQueueImpl, getSyclObjImpl(Queue));
}

// Regression test for infinite loop previously in tryReuseExistingAllocation
// in the graph memory pool implementation. When an unusable node
// from the dependency graph of a malloc node is visited twice in the search,
// then the implementation infinitely looped over this node. This occurs
// in the last malloc call below, where the diamond dependency caused by E1 & E2
// cause the allocation associated with FreeEvent to be visited twice in the
// search.
TEST_F(CommandGraphTest, AsyncAllocInfiniteLoop) {
  const size_t SmallSize = 1 << 16; // 64KB
  const size_t BigSize = 1 << 17;   // 128KB

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Queue.get_context(), Dev};

  Graph.begin_recording(Queue);

  void *Ptr1 = nullptr;
  void *Ptr2 = nullptr;
  void *Ptr3 = nullptr;

  // An unreachable allocation of SmallSize to trigger the dependency node
  // search on the next allocation of SmallSize.
  Queue.submit([&](handler &CGH) {
    Ptr1 = experimental::async_malloc(CGH, usm::alloc::device, SmallSize);
  });
  Queue.submit([&](handler &CGH) { experimental::async_free(CGH, Ptr1); });

  // Incompatible allocation that will get visited twice.
  Queue.submit([&](handler &CGH) {
    Ptr2 = experimental::async_malloc(CGH, usm::alloc::device, BigSize);
  });
  auto FreeEvent =
      Queue.submit([&](handler &CGH) { experimental::async_free(CGH, Ptr2); });

  // Create diamond dependency to last allocation
  auto E1 = Queue.submit([&](handler &CGH) { CGH.depends_on(FreeEvent); });
  auto E2 = Queue.submit([&](handler &CGH) { CGH.depends_on(FreeEvent); });

  // Triggers infinite loop prior to fix
  Queue.submit([&](handler &CGH) {
    CGH.depends_on({E1, E2});
    Ptr3 = experimental::async_malloc(CGH, usm::alloc::device, SmallSize);
  });
  Queue.submit([&](handler &CGH) { experimental::async_free(CGH, Ptr3); });

  Graph.end_recording(Queue);
}

// Regression test for a node being silently dropped from the command-buffer
// schedule. When a host task partitions the graph, the topological sort per
// partition only traverses edges internal to that partition, but it used to
// wait for all predecessors (including cross-partition ones) to be visited
// before scheduling a node. A node with a cross-partition predecessor could
// therefore never reach its completion threshold and, along with everything
// downstream of it, was omitted from the schedule and never enqueued.
//
// The topology below reproduces the issue with two in-order queues recording
// into one graph and barriers creating the cross-queue edges:
//
//   Q2 chain:  BQ2a ---------> BQ2b --> BQ2c
//              |               ^        |
//              |[A]            |[B]     |[C]  (cross-queue barrier edges)
//              v               |        v
//   Q1 chain:  BQ1a --> HT --> BQ1b --> BQ1c --> K
//
// The host task HT splits the graph. BQ2a is pulled into HT's predecessor
// partition, but its in-order successor BQ2b stays in the successor partition,
// creating the cross-partition edge BQ2a -> BQ2b that lands on an interior
// (non-root) node. Prior to the fix, BQ2b (and hence BQ2c, BQ1c and the kernel
// K) were dropped from the schedule.
TEST_F(CommandGraphTest, HostTaskPartitionCrossQueueBarrier) {
  sycl::property_list InOrder{sycl::property::queue::in_order()};
  sycl::queue Q1{Queue.get_context(), Dev, InOrder};
  sycl::queue Q2{Queue.get_context(), Dev, InOrder};

  experimental::command_graph<experimental::graph_state::modifiable> G{
      Q1.get_context(), Q1.get_device()};

  G.begin_recording({Q1, Q2});

  // [A] pre-HT: Q2 -> Q1
  {
    sycl::event E = Q2.ext_oneapi_submit_barrier();
    Q1.ext_oneapi_submit_barrier({E});
  }

  // Host task on Q1, which forces graph partitioning.
  Q1.submit([&](sycl::handler &CGH) { CGH.host_task([]() {}); });

  // [B] post-HT: Q1 -> Q2
  {
    sycl::event E = Q1.ext_oneapi_submit_barrier();
    Q2.ext_oneapi_submit_barrier({E});
  }

  // [C] pre-K: Q2 -> Q1
  {
    sycl::event E = Q2.ext_oneapi_submit_barrier();
    Q1.ext_oneapi_submit_barrier({E});
  }

  G.end_recording(Q2);

  // Kernel K on Q1, downstream of the barrier chain. This is the node that was
  // silently dropped prior to the fix.
  auto KernelEvent = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  G.end_recording();

  experimental::detail::graph_impl &GraphImpl = *getSyclObjImpl(G);
  experimental::detail::node_impl &KernelNode =
      GraphImpl.getNodeForEvent(getSyclObjImpl(KernelEvent));

  auto GraphExec = G.finalize();
  experimental::detail::exec_graph_impl &ExecGraphImpl =
      *getSyclObjImpl(GraphExec);

  // Every node that requires enqueue must appear in the schedule. The kernel is
  // the node that was dropped; assert on it explicitly, and also check that no
  // enqueue-requiring node is missing overall.
  const auto &Schedule = ExecGraphImpl.getSchedule();

  size_t NumEnqueuedNodesInGraph = 0;
  for (experimental::detail::node_impl &Node : GraphImpl.nodes()) {
    if (Node.requiresEnqueue())
      ++NumEnqueuedNodesInGraph;
  }

  size_t NumEnqueuedNodesInSchedule = 0;
  bool KernelFoundInSchedule = false;
  for (experimental::detail::node_impl *Node : Schedule) {
    if (Node->requiresEnqueue())
      ++NumEnqueuedNodesInSchedule;
    if (Node->isSimilar(KernelNode))
      KernelFoundInSchedule = true;
  }

  ASSERT_TRUE(KernelFoundInSchedule);
  ASSERT_EQ(NumEnqueuedNodesInSchedule, NumEnqueuedNodesInGraph);
}
