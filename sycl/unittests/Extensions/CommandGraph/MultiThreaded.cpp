//==---------------------- MultiThreaded.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

// Test Fixture
class MultiThreadGraphTest : public CommandGraphTest {
public:
  MultiThreadGraphTest()
      : CommandGraphTest(), NumThreads(std::thread::hardware_concurrency()),
        SyncPoint(NumThreads) {
    Threads.reserve(NumThreads);
  }

protected:
  const unsigned NumThreads;
  Barrier SyncPoint;
  std::vector<std::thread> Threads;
};

// anonymous namespace used to avoid code redundancy by defining functions
// used by multiple times by unitests.
// Defining anonymous namespace prevents from function naming conflits
namespace {
/// Submits four kernels with diamond dependency to the queue Q
/// @param Q Queue to submit nodes to.
void runKernels(queue Q) {
  auto NodeA = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(NodeA);
    cgh.single_task<TestKernel<>>([]() {});
  });
  auto NodeC = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(NodeA);
    cgh.single_task<TestKernel<>>([]() {});
  });
  auto NodeD = Q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({NodeB, NodeC});
    cgh.single_task<TestKernel<>>([]() {});
  });
}

/// Submits four kernels without any additional dependencies the queue Q
/// @param Q Queue to submit nodes to.
void runKernelsInOrder(queue Q) {
  auto NodeA = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeC = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeD = Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
}

/// Adds four kernels with diamond dependency to the Graph G
/// @param G Modifiable graph to add commands to.
void addKernels(
    experimental::command_graph<experimental::graph_state::modifiable> G) {
  auto NodeA = G.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto NodeB =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeA)});
  auto NodeC =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeA)});
  auto NodeD =
      G.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
            {experimental::property::node::depends_on(NodeB, NodeC)});
}

bool checkExecGraphSchedule(
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphA,
    std::shared_ptr<sycl::ext::oneapi::experimental::detail::exec_graph_impl>
        GraphB) {
  auto ScheduleA = GraphA->getSchedule();
  auto ScheduleB = GraphB->getSchedule();
  if (ScheduleA.size() != ScheduleB.size())
    return false;

  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleA{std::begin(ScheduleA), std::end(ScheduleA)};
  std::vector<
      std::shared_ptr<sycl::ext::oneapi::experimental::detail::node_impl>>
      VScheduleB{std::begin(ScheduleB), std::end(ScheduleB)};

  for (size_t i = 0; i < VScheduleA.size(); i++) {
    if (!VScheduleA[i]->isSimilar(VScheduleB[i]))
      return false;
  }
  return true;
}
} // namespace

TEST_F(MultiThreadGraphTest, BeginEndRecording) {
  auto RecordGraph = [&]() {
    queue MyQueue{Queue.get_context(), Queue.get_device()};

    SyncPoint.wait();

    Graph.begin_recording(MyQueue);
    runKernels(MyQueue);
    Graph.end_recording(MyQueue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef{Queue.get_context(), Queue.get_device()};
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  for (unsigned i = 0; i < NumThreads; ++i) {
    queue MyQueue{Queue.get_context(), Queue.get_device()};
    GraphRef.begin_recording(MyQueue);
    runKernels(MyQueue);
    GraphRef.end_recording(MyQueue);
  }

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, ExplicitAddNodes) {
  auto RecordGraph = [&]() {
    queue MyQueue{Queue.get_context(), Queue.get_device()};

    SyncPoint.wait();
    addKernels(Graph);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef;
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  for (unsigned i = 0; i < NumThreads; ++i) {
    addKernels(GraphRef);
  }

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, RecordAddNodes) {
  Graph.begin_recording(Queue);
  auto RecordGraph = [&]() {
    SyncPoint.wait();
    runKernels(Queue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // We stop recording the Queue when all threads have finished their processing
  Graph.end_recording(Queue);

  // Reference computation
  queue QueueRef{Queue.get_context(), Queue.get_device()};
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  GraphRef.begin_recording(QueueRef);
  for (unsigned i = 0; i < NumThreads; ++i) {
    runKernels(QueueRef);
  }
  GraphRef.end_recording(QueueRef);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(GraphRef);
  ASSERT_EQ(GraphImpl->hasSimilarStructure(GraphRefImpl), true);
}

TEST_F(MultiThreadGraphTest, RecordAddNodesInOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  queue InOrderQueue{Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraph{InOrderQueue.get_context(), InOrderQueue.get_device()};

  InOrderGraph.begin_recording(InOrderQueue);
  auto RecordGraph = [&]() {
    SyncPoint.wait();
    runKernelsInOrder(InOrderQueue);
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // We stop recording the Queue when all threads have finished their processing
  InOrderGraph.end_recording(InOrderQueue);

  // Reference computation
  queue InOrderQueueRef{Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable>
      InOrderGraphRef{InOrderQueueRef.get_context(),
                      InOrderQueueRef.get_device()};

  InOrderGraphRef.begin_recording(InOrderQueueRef);
  for (unsigned i = 0; i < NumThreads; ++i) {
    runKernelsInOrder(InOrderQueueRef);
  }
  InOrderGraphRef.end_recording(InOrderQueueRef);

  auto GraphImpl = sycl::detail::getSyclObjImpl(InOrderGraph);
  auto GraphRefImpl = sycl::detail::getSyclObjImpl(InOrderGraphRef);
  ASSERT_EQ(GraphImpl->getNumberOfNodes(), GraphRefImpl->getNumberOfNodes());

  // In-order graph must have only a single root
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);

  // Check structure graph
  auto CurrentNode = (*GraphImpl->MRoots.begin()).lock();
  for (size_t i = 1; i <= GraphImpl->getNumberOfNodes(); i++) {
    EXPECT_LE(CurrentNode->MSuccessors.size(), 1lu);

    // Checking the last node has no successors
    if (i == GraphImpl->getNumberOfNodes()) {
      EXPECT_EQ(CurrentNode->MSuccessors.size(), 0lu);
    } else {
      // Check other nodes have 1 successor
      EXPECT_EQ(CurrentNode->MSuccessors.size(), 1lu);
      CurrentNode = CurrentNode->MSuccessors[0].lock();
    }
  }
}

TEST_F(MultiThreadGraphTest, Finalize) {
  addKernels(Graph);

  std::mutex MutexMap;

  std::map<int,
           experimental::command_graph<experimental::graph_state::executable>>
      GraphsExecMap;
  auto FinalizeGraph = [&](int ThreadNum) {
    SyncPoint.wait();
    auto GraphExec = Graph.finalize();
    Queue.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });

    std::lock_guard<std::mutex> Guard(MutexMap);
    GraphsExecMap.insert(
        std::map<int, experimental::command_graph<
                          experimental::graph_state::executable>>::
            value_type(ThreadNum, GraphExec));
  };

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(FinalizeGraph, i);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Reference computation
  queue QueueRef;
  experimental::command_graph<experimental::graph_state::modifiable> GraphRef{
      Queue.get_context(), Queue.get_device()};

  addKernels(GraphRef);

  for (unsigned i = 0; i < NumThreads; ++i) {
    auto GraphExecRef = GraphRef.finalize();
    QueueRef.submit(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExecRef); });
    auto GraphExecImpl =
        sycl::detail::getSyclObjImpl(GraphsExecMap.find(i)->second);
    auto GraphExecRefImpl = sycl::detail::getSyclObjImpl(GraphExecRef);
    ASSERT_EQ(checkExecGraphSchedule(GraphExecImpl, GraphExecRefImpl), true);
  }
}
