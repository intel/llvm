//==-------------------------- GetInfo.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests command_graph::get_info() for graphs managed by the SYCL runtime.

#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

TEST_F(CommandGraphTest, GetInfoRecordingModeRuntime) {
  Graph.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  EXPECT_EQ(Graph.get_info<experimental::info::graph::recording_mode>(),
            experimental::graph_recording_mode::runtime);

  auto ExecGraph = Graph.finalize();
  EXPECT_EQ(ExecGraph.get_info<experimental::info::graph::recording_mode>(),
            experimental::graph_recording_mode::runtime);
}

TEST_F(CommandGraphTest, GetInfoUpdatable) {
  Graph.add([&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  // Finalizing the same modifiable graph with and without the property
  auto ExecGraph = Graph.finalize();
  auto UpdatableExecGraph =
      Graph.finalize(experimental::property::graph::updatable{});

  EXPECT_FALSE(ExecGraph.get_info<experimental::info::graph::updatable>());
  EXPECT_TRUE(
      UpdatableExecGraph.get_info<experimental::info::graph::updatable>());
}
