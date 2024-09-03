//==------------------------- Exceptions.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

#if SYCL_EXT_CODEPLAY_KERNEL_FUSION

// Various tests which are checking for correct exception behaviour
// for Graph Fusion.

TEST_F(CommandGraphTest, GraphFusionMultiPatitionsException) {

  queue QueueInOrder{sycl::property::queue::in_order{}};
  {
    experimental::command_graph Graph{
        QueueInOrder.get_context(), QueueInOrder.get_device(),
        experimental::property::graph::assume_buffer_outlives_graph{}};

    Graph.begin_recording(QueueInOrder);

    QueueInOrder.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

    // Add a host task in the middle of the graph
    QueueInOrder.submit([&](handler &CGH) { CGH.host_task([=]() {}); });

    QueueInOrder.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

    Graph.end_recording();

    // Check that no exception is throw if fusion is enabled
    bool Success = true;
    try {
      auto ExecGraph =
          Graph.finalize({experimental::property::graph::enable_fusion{}});
    } catch (sycl::exception &Exception) {
      Success = false;
    }
    ASSERT_EQ(Success, true);

    // Check exception if fusion is required
    try {
      auto ExecGraph =
          Graph.finalize({experimental::property::graph::require_fusion{}});
    } catch (sycl::exception &Exception) {
      Success = false;
    }
    ASSERT_EQ(Success, false);
  }
}

TEST_F(CommandGraphTest, GraphFusionMemoryCmdException) {

  constexpr size_t DataSize = 512;
  std::vector<int> DataA(DataSize);
  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataA};
  BufferB.set_write_back(false);

  queue QueueInOrder{sycl::property::queue::in_order{}};
  {
    experimental::command_graph Graph{
        QueueInOrder.get_context(), QueueInOrder.get_device(),
        experimental::property::graph::assume_buffer_outlives_graph{}};

    Graph.begin_recording(QueueInOrder);

    QueueInOrder.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

    QueueInOrder.submit([&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      CGH.copy(AccB, AccA);
    });

    QueueInOrder.submit(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });

    Graph.end_recording();

    // Check that no exception is throw if fusion is enabled
    bool Success = true;
    try {
      auto ExecGraph =
          Graph.finalize({experimental::property::graph::enable_fusion{}});
    } catch (sycl::exception &Exception) {
      Success = false;
    }
    ASSERT_EQ(Success, true);

    // Check exception if fusion is required
    try {
      auto ExecGraph =
          Graph.finalize({experimental::property::graph::require_fusion{}});
    } catch (sycl::exception &Exception) {
      Success = false;
    }
    ASSERT_EQ(Success, false);
  }
}

#endif
