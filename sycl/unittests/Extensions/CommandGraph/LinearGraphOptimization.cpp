//==--------------------- LinearGraphOptimization.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for linear graph optimization which skips creating and tracking UR sync
// points. Optimization is an internal implementation detail, validated through
// inspecting private members of exec_graph_impl. Test achieves two goals: 1)
// Validates that linear partitions in graphs are optimized to avoid using UR
// sync points 2) Validates that non-linear partitions contain the expected
// number of sync points

#include "Common.hpp"
#include <optional>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::detail;

// Helper to build a linear chain of N kernels on a queue inside graph capture.
static void BuildLinearChain(queue &Queue, bool IsInOrderQueue, int N) {
  std::optional<sycl::event> Event;
  for (int I = 0; I < N; ++I) {
    if (IsInOrderQueue) {
      experimental::single_task<TestKernel>(Queue, []() {});
    } else {
      Event = Queue.submit([&](handler &h) {
        if (Event) {
          h.depends_on(*Event);
        }
        h.single_task<TestKernel>([]() {});
      });
    }
  }
}

// Validate linear optimization invariants on an executable graph.
static void ValidateLinearExec(exec_graph_impl &Impl, int NumLinearChains) {
  EXPECT_EQ(GraphImplTest::NumPartitionsInOrder(Impl), NumLinearChains);
  EXPECT_EQ(GraphImplTest::NumSyncPoints(Impl), 0);
}

TEST_F(CommandGraphTest, LinearInOrderQueue) {
  sycl::property_list Props{sycl::property::queue::in_order{}};
  queue InOrderQ{Dev, Props};

  experimental::command_graph<graph_state::modifiable> G{InOrderQ.get_context(),
                                                         InOrderQ.get_device()};
  G.begin_recording(InOrderQ);
  BuildLinearChain(InOrderQ, /*IsInOrderQueue=*/true, /*N=*/3);
  InOrderQ.submit([&](sycl::handler &cgh) { cgh.host_task([]() {}); });
  BuildLinearChain(InOrderQ, /*IsInOrderQueue=*/true, /*N=*/4);
  G.end_recording(InOrderQ);

  auto Exec = G.finalize();
  auto &Impl = *getSyclObjImpl(Exec);
  ValidateLinearExec(Impl, /*InOrderPartitions=*/3);
}

TEST_F(CommandGraphTest, LinearOutOfOrderQueue) {
  // Out-of-order queue but we submit a strict linear dependency chain by
  // adding explicit depends_on between each node to achieve linearity.
  queue OOOQ{Dev};
  experimental::command_graph<graph_state::modifiable> G{OOOQ.get_context(),
                                                         OOOQ.get_device()};
  G.begin_recording(OOOQ);
  BuildLinearChain(OOOQ, /*IsInOrderQueue=*/false, /*N=*/6);
  G.end_recording(OOOQ);

  auto Exec = G.finalize();
  auto &Impl = *getSyclObjImpl(Exec);
  ValidateLinearExec(Impl, /*InOrderPartitions=*/1);
}

// Ensures non-linear graphs are creating and tracking sync points internally
// for proper scheduling and that the linear optimization is not improperly
// applied.
TEST_F(CommandGraphTest, NonLinearOutOfOrderQueue) {
  queue Q{Dev};
  experimental::command_graph<graph_state::modifiable> G{Q.get_context(),
                                                         Q.get_device()};
  G.begin_recording(Q);
  // Root node
  event Root = Q.submit([&](handler &h) { h.single_task<TestKernel>([] {}); });
  // Two parallel branches depending on Root
  event A = Q.submit([&](handler &h) {
    h.depends_on(Root);
    h.single_task<TestKernel>([] {});
  });
  event B = Q.submit([&](handler &h) {
    h.depends_on(Root);
    h.single_task<TestKernel>([] {});
  });
  // Join node depends on both A and B
  Q.submit([&](handler &h) {
    h.depends_on(A);
    h.depends_on(B);
    h.single_task<TestKernel>([] {});
  });
  G.end_recording(Q);

  auto Exec = G.finalize();
  auto &Impl = *getSyclObjImpl(Exec);

  const int NumLinear = GraphImplTest::NumPartitionsInOrder(Impl);
  const int NumSyncPoints = GraphImplTest::NumSyncPoints(Impl);

  // We should track a sync point per node for a total of 4
  EXPECT_EQ(NumSyncPoints, 4);
  EXPECT_EQ(NumLinear, 0);
}
