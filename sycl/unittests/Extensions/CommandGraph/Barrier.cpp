//==---------------------------- Barrier.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, EnqueueBarrier) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier =
      Queue.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (B)
  //     / \
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto BarrierNode = Node->MSuccessors.front().lock();
    ASSERT_EQ(BarrierNode->MCGType, sycl::detail::CG::Barrier);
    ASSERT_EQ(GraphImpl->getEventForNode(BarrierNode),
              sycl::detail::getSyclObjImpl(Barrier));
    ASSERT_EQ(BarrierNode->MPredecessors.size(), 3lu);
    ASSERT_EQ(BarrierNode->MSuccessors.size(), 2lu);
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierMultipleQueues) {
  sycl::queue Queue2{Queue.get_context(), Dev};
  Graph.begin_recording({Queue, Queue2});
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier =
      Queue2.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node4Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |  /
  //    \ | /
  //     (B)
  //     / \
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto BarrierNode = Node->MSuccessors.front().lock();
    ASSERT_EQ(BarrierNode->MCGType, sycl::detail::CG::Barrier);
    ASSERT_EQ(GraphImpl->getEventForNode(BarrierNode),
              sycl::detail::getSyclObjImpl(Barrier));
    ASSERT_EQ(BarrierNode->MPredecessors.size(), 3lu);
    ASSERT_EQ(BarrierNode->MSuccessors.size(), 2lu);
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierWaitList) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //     (B)  |
  //     / \ /
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 2lu);
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
    }
  }
}

TEST_F(CommandGraphTest, EnqueueBarrierWaitListMultipleQueues) {
  sycl::queue Queue2{Queue.get_context(), Dev};
  Graph.begin_recording({Queue, Queue2});
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  // Node1Graph comes from Queue, and Node2Graph comes from Queue2
  auto Barrier = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  auto Barrier2 = Queue2.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Barrier, Node4Graph, Node5Graph});
  });

  Graph.end_recording();

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //     (B)  |
  //     /|\ /
  //   (4)|(5)
  //    \ | /
  //     \|/
  //     (B2)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 3lu);
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
    }
  }
}

TEST_F(CommandGraphTest, EnqueueMultipleBarrier) {
  Graph.begin_recording(Queue);
  auto Node1Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node3Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Barrier1 = Queue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_barrier({Node1Graph, Node2Graph});
  });

  auto Node4Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node3Graph);
    cgh.single_task<TestKernel<>>([]() {});
  });

  auto Barrier2 =
      Queue.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_barrier(); });

  auto Node6Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node7Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node8Graph = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording(Queue);

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //   \  |   |
  //    \ |   |
  //    (B1)  |
  //     /|\ /
  //   (4)|(5)
  //     \|/
  //     (B2)
  //     /|\
  //    / | \
  // (6) (7) (8)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto Node = Root.lock();
    ASSERT_EQ(Node->MSuccessors.size(), 1lu);
    auto SuccNode = Node->MSuccessors.front().lock();
    if (SuccNode->MCGType == sycl::detail::CG::Barrier) {
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier1));
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 3lu);
      for (auto Succ1 : SuccNode->MSuccessors) {
        auto SuccBarrier1 = Succ1.lock();
        if (SuccBarrier1->MCGType == sycl::detail::CG::Barrier) {
          ASSERT_EQ(GraphImpl->getEventForNode(SuccBarrier1),
                    sycl::detail::getSyclObjImpl(Barrier2));
          ASSERT_EQ(SuccBarrier1->MPredecessors.size(), 3lu);
          ASSERT_EQ(SuccBarrier1->MSuccessors.size(), 3lu);
          for (auto Succ2 : SuccBarrier1->MSuccessors) {
            auto SuccBarrier2 = Succ2.lock();
            // Nodes 6, 7, 8
            ASSERT_EQ(SuccBarrier2->MPredecessors.size(), 1lu);
            ASSERT_EQ(SuccBarrier2->MSuccessors.size(), 0lu);
          }
        } else {
          // Node 4 or Node 5
          if (GraphImpl->getEventForNode(SuccBarrier1) ==
              sycl::detail::getSyclObjImpl(Node4Graph)) {
            // Node 4
            ASSERT_EQ(SuccBarrier1->MPredecessors.size(), 1lu);
            ASSERT_EQ(SuccBarrier1->MSuccessors.size(), 1lu);
          }
        }
      }
    } else {
      // Node 5
      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 1lu);
    }
  }
}
