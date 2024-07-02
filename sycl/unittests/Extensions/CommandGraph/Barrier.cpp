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

  auto Barrier = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.ext_oneapi_barrier({Node2Graph}); });

  auto Node4Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node5Graph = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  // Check the graph structure
  // (1) (2) (3)
  //      |
  //     (B)
  //     / \
  //   (4) (5)
  ASSERT_EQ(GraphImpl->MRoots.size(), 3lu);
  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(Node2Graph)) {

      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);
      auto SuccNode = RootNode->MSuccessors.front().lock();

      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 2lu);

      for (auto SuccSucc : SuccNode->MSuccessors) {
        auto SuccSuccNode = SuccSucc.lock();

        if (GraphImpl->getEventForNode(SuccSuccNode) ==
            sycl::detail::getSyclObjImpl(Node4Graph)) {
          ASSERT_EQ(SuccSuccNode->MPredecessors.size(), 1lu);
          ASSERT_EQ(SuccSuccNode->MSuccessors.size(), 0lu);
        } else if (GraphImpl->getEventForNode(SuccSuccNode) ==
                   sycl::detail::getSyclObjImpl(Node5Graph)) {
          ASSERT_EQ(SuccSuccNode->MPredecessors.size(), 1lu);
          ASSERT_EQ(SuccSuccNode->MSuccessors.size(), 0lu);
        } else {
          ASSERT_TRUE(false && "Unexpected node");
        }
      }
    } else {
      ASSERT_EQ(RootNode->MSuccessors.size(), 0lu);
    }
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

TEST_F(CommandGraphTest, InOrderQueueWithPreviousCommand) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};

  auto NonGraphEvent = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.begin_recording(InOrderQueue);

  ASSERT_THROW(
      {
        try {
          InOrderQueue.ext_oneapi_submit_barrier({NonGraphEvent});
        } catch (const sycl::exception &e) {
          ASSERT_EQ(e.code(), make_error_code(sycl::errc::invalid));
          throw;
        }
      },
      sycl::exception);

  InOrderQueue.ext_oneapi_submit_barrier();
  Graph.end_recording(InOrderQueue);

  // Check the graph structure
  // (B)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 1lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    ASSERT_EQ(RootNode->MSuccessors.size(), 0lu);
    ASSERT_TRUE(RootNode->MCGType == sycl::detail::CG::Barrier);
  }
}

TEST_F(CommandGraphTest, InOrderQueuesWithBarrier) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue1{Dev, Properties};
  sycl::queue InOrderQueue2{InOrderQueue1.get_context(), Dev, Properties};
  sycl::queue InOrderQueue3{InOrderQueue1.get_context(), Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue1};

  Graph.begin_recording({InOrderQueue1, InOrderQueue2, InOrderQueue3});

  auto Node1 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  InOrderQueue3.ext_oneapi_submit_barrier({Node1});

  Graph.end_recording();

  // Check the graph structure
  // (1) (2)
  //  |
  // (B)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(Node1)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);

      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_TRUE(SuccNode->MCGType == sycl::detail::CG::Barrier);

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else if (GraphImpl->getEventForNode(RootNode) ==
               sycl::detail::getSyclObjImpl(Node2)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false && "Unexpected root node");
    }
  }
}

TEST_F(CommandGraphTest, InOrderQueuesWithBarrierWaitList) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue1{Dev, Properties};
  sycl::queue InOrderQueue2{InOrderQueue1.get_context(), Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue1};

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});

  auto Node1 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = InOrderQueue2.ext_oneapi_submit_barrier({Node1});

  Graph.end_recording();

  // Check the graph structure
  // (1) (2)
  //  |  /
  // (B)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);

    auto SuccNode = RootNode->MSuccessors.front().lock();
    ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
              sycl::detail::getSyclObjImpl(BarrierNode));

    ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
    ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
  }
}

TEST_F(CommandGraphTest, InOrderQueuesWithEmptyBarrierWaitList) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue1{Dev, Properties};
  sycl::queue InOrderQueue2{InOrderQueue1.get_context(), Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue1};

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});

  auto Node1 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = InOrderQueue1.ext_oneapi_submit_barrier();

  auto Node3 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording();

  // Check the graph structure
  // (1)  (2)
  //  |    |
  // (B)  (3)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(Node1)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);

      auto SuccNode = RootNode->MSuccessors.front().lock();

      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(BarrierNode));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else if (GraphImpl->getEventForNode(RootNode) ==
               sycl::detail::getSyclObjImpl(Node2)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);

      auto SuccNode = RootNode->MSuccessors.front().lock();

      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Node3));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false && "Unexpected root node");
    }
  }
}

TEST_F(CommandGraphTest, BarrierMixedQueueTypes) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};
  sycl::queue OutOfOrderQueue{InOrderQueue.get_context(), Dev};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue};

  Graph.begin_recording({InOrderQueue, OutOfOrderQueue});

  auto Node1 = OutOfOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = OutOfOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = InOrderQueue.ext_oneapi_submit_barrier({Node1, Node2});

  auto Node3 = OutOfOrderQueue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(Node2);
    cgh.single_task<TestKernel<>>([]() {});
  });

  Graph.end_recording();

  // Check the graph structure
  // (1)  (2)
  //  \   /|
  //   (B) |
  //       |
  //      (3)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(Node1)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);
    } else if (GraphImpl->getEventForNode(RootNode) ==
               sycl::detail::getSyclObjImpl(Node2)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 2lu);
    } else {
      ASSERT_TRUE(false && "Unexpected root node");
    }

    for (auto Succ : RootNode->MSuccessors) {
      auto SuccNode = Succ.lock();

      if (GraphImpl->getEventForNode(SuccNode) ==
          sycl::detail::getSyclObjImpl(BarrierNode)) {
        ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
        ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
      } else if (GraphImpl->getEventForNode(SuccNode) ==
                 sycl::detail::getSyclObjImpl(Node3)) {
        ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
        ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
      } else {
        ASSERT_TRUE(false && "Unexpected root node");
      }
    }
  }
}

TEST_F(CommandGraphTest, BarrierBetweenExplicitNodes) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Dev, Properties};

  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue};

  auto Node1 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.begin_recording(InOrderQueue);
  auto BarrierNode = InOrderQueue.ext_oneapi_submit_barrier();
  Graph.end_recording();

  auto Node2 = Graph.add(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
      {experimental::property::node::depends_on(Node1)});

  // Check the graph structure
  // (B) (1)
  //      |
  //     (2)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();

    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(BarrierNode)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 0lu);
    } else if (RootNode == sycl::detail::getSyclObjImpl(Node1)) {
      ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(SuccNode, sycl::detail::getSyclObjImpl(Node2));
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(CommandGraphTest, BarrierMultipleOOOQueue) {
  sycl::queue Queue2{Queue.get_context(), Dev};
  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Queue};

  Graph.begin_recording({Queue, Queue2});

  auto Node1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node3 = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node4 = Queue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = Queue.ext_oneapi_submit_barrier();

  auto Node5 = Queue2.submit([&](sycl::handler &cgh) {
    cgh.depends_on({Node3, Node4});
    cgh.single_task<TestKernel<>>([]() {});
  });

  auto Node6 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording();

  // Check the graph structure
  // (1) (2) (3) (4)
  //  \  /     \ /
  //  (B)      (5)
  //   |
  //  (6)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 4u);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    auto RootNodeEvent = GraphImpl->getEventForNode(RootNode);
    if ((RootNodeEvent == sycl::detail::getSyclObjImpl(Node1)) ||
        (RootNodeEvent == sycl::detail::getSyclObjImpl(Node2))) {

      auto SuccNode = RootNode->MSuccessors.front().lock();

      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(BarrierNode));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 1lu);

      auto SuccSuccNode = SuccNode->MSuccessors.front().lock();

      ASSERT_EQ(SuccSuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccSuccNode->MSuccessors.size(), 0lu);

      auto Node6Impl = sycl::detail::getSyclObjImpl(Node6);
      ASSERT_EQ(GraphImpl->getEventForNode(SuccSuccNode), Node6Impl);
    } else if ((RootNodeEvent == sycl::detail::getSyclObjImpl(Node3)) ||
               (RootNodeEvent == sycl::detail::getSyclObjImpl(Node4))) {
      auto SuccNode = RootNode->MSuccessors.front().lock();

      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Node5));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 2lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(CommandGraphTest, BarrierMultipleInOrderQueue) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue1{Queue.get_context(), Dev, Properties};
  sycl::queue InOrderQueue2{Queue.get_context(), Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      InOrderQueue1};

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});

  auto Node1 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = InOrderQueue1.ext_oneapi_submit_barrier();

  auto Node3 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording();

  // Check the graph structure
  // (1) (2)
  //  |   |
  // (B) (3)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2u);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    auto RootNodeEvent = GraphImpl->getEventForNode(RootNode);
    if (RootNodeEvent == sycl::detail::getSyclObjImpl(Node1)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(BarrierNode));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else if (RootNodeEvent == sycl::detail::getSyclObjImpl(Node2)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Node3));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(CommandGraphTest, BarrierMultipleMixedOrderQueues) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Queue.get_context(), Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Queue};

  Graph.begin_recording({Queue, InOrderQueue});

  auto Node1 = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto Node2 = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  auto BarrierNode = Queue.ext_oneapi_submit_barrier();

  auto Node3 = InOrderQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

  Graph.end_recording();

  // Check the graph structure
  // (1) (2)
  //  |   |
  // (B) (3)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2u);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    auto RootNodeEvent = GraphImpl->getEventForNode(RootNode);
    if (RootNodeEvent == sycl::detail::getSyclObjImpl(Node1)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(BarrierNode));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else if (RootNodeEvent == sycl::detail::getSyclObjImpl(Node2)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Node3));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(CommandGraphTest, BarrierMultipleQueuesMultipleBarriers) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue{Queue.get_context(), Dev, Properties};
  experimental::command_graph<experimental::graph_state::modifiable> Graph{
      Queue};

  Graph.begin_recording({Queue, InOrderQueue});

  auto Barrier1 = Queue.ext_oneapi_submit_barrier();
  auto Barrier2 = InOrderQueue.ext_oneapi_submit_barrier();
  auto Barrier3 = InOrderQueue.ext_oneapi_submit_barrier();
  auto Barrier4 = Queue.ext_oneapi_submit_barrier();

  Graph.end_recording();

  // Check the graph structure
  // (1)       (2)
  //  |         |
  // (4)       (3)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2u);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    auto RootNodeEvent = GraphImpl->getEventForNode(RootNode);
    if (RootNodeEvent == sycl::detail::getSyclObjImpl(Barrier1)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier4));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else if (RootNodeEvent == sycl::detail::getSyclObjImpl(Barrier2)) {
      auto SuccNode = RootNode->MSuccessors.front().lock();
      ASSERT_EQ(GraphImpl->getEventForNode(SuccNode),
                sycl::detail::getSyclObjImpl(Barrier3));

      ASSERT_EQ(SuccNode->MPredecessors.size(), 1lu);
      ASSERT_EQ(SuccNode->MSuccessors.size(), 0lu);
    } else {
      ASSERT_TRUE(false);
    }
  }
}

TEST_F(CommandGraphTest, BarrierWithInOrderCommands) {
  sycl::property_list Properties{sycl::property::queue::in_order()};
  sycl::queue InOrderQueue1{Dev, Properties};
  sycl::queue InOrderQueue2{Dev, Properties};

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Node1 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node2 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Barrier1 = InOrderQueue1.ext_oneapi_submit_barrier();
  auto Barrier2 = InOrderQueue2.ext_oneapi_submit_barrier();
  Graph.end_recording();

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Node3 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node4 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Barrier3 = InOrderQueue1.ext_oneapi_submit_barrier();
  auto Barrier4 = InOrderQueue2.ext_oneapi_submit_barrier();
  Graph.end_recording();

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Node5 = InOrderQueue1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  auto Node6 = InOrderQueue2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  Graph.end_recording();

  Graph.begin_recording({InOrderQueue1, InOrderQueue2});
  auto Barrier5 = InOrderQueue1.ext_oneapi_submit_barrier({Node5, Node6});
  Graph.end_recording();

  // Check the graph structure
  // (1)    (2)
  //  |      |
  // (B1)   (B2)
  //  |      |
  // (3)    (4)
  //  |      |
  // (B3)   (B4)
  //  |      |
  // (5)    (6)
  //    \   /
  //    (B5)
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);
  ASSERT_EQ(GraphImpl->MRoots.size(), 2lu);

  for (auto Root : GraphImpl->MRoots) {
    auto RootNode = Root.lock();
    bool EvenPath;

    ASSERT_EQ(RootNode->MSuccessors.size(), 1lu);
    if (GraphImpl->getEventForNode(RootNode) ==
        sycl::detail::getSyclObjImpl(Node2)) {
      EvenPath = true;
    } else if (GraphImpl->getEventForNode(RootNode),
               sycl::detail::getSyclObjImpl(Node1)) {
      EvenPath = false;
    } else {
      ASSERT_TRUE(false);
    }

    auto Succ1Node = RootNode->MSuccessors.front().lock();
    ASSERT_EQ(Succ1Node->MSuccessors.size(), 1lu);
    if (EvenPath) {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ1Node),
                sycl::detail::getSyclObjImpl(Barrier2));
    } else {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ1Node),
                sycl::detail::getSyclObjImpl(Barrier1));
    }

    auto Succ2Node = Succ1Node->MSuccessors.front().lock();
    ASSERT_EQ(Succ2Node->MSuccessors.size(), 1lu);
    if (EvenPath) {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ2Node),
                sycl::detail::getSyclObjImpl(Node4));
    } else {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ2Node),
                sycl::detail::getSyclObjImpl(Node3));
    }

    auto Succ3Node = Succ2Node->MSuccessors.front().lock();
    ASSERT_EQ(Succ3Node->MSuccessors.size(), 1lu);
    if (EvenPath) {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ3Node),
                sycl::detail::getSyclObjImpl(Barrier4));
    } else {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ3Node),
                sycl::detail::getSyclObjImpl(Barrier3));
    }

    auto Succ4Node = Succ3Node->MSuccessors.front().lock();
    ASSERT_EQ(Succ4Node->MSuccessors.size(), 1lu);
    if (EvenPath) {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ4Node),
                sycl::detail::getSyclObjImpl(Node6));
    } else {
      ASSERT_EQ(GraphImpl->getEventForNode(Succ4Node),
                sycl::detail::getSyclObjImpl(Node5));
    }

    auto Succ5Node = Succ4Node->MSuccessors.front().lock();
    ASSERT_EQ(Succ5Node->MSuccessors.size(), 0lu);
    ASSERT_EQ(Succ5Node->MPredecessors.size(), 2lu);
    ASSERT_EQ(GraphImpl->getEventForNode(Succ5Node),
              sycl::detail::getSyclObjImpl(Barrier5));
  }
}
