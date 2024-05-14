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
  Graph.begin_recording(Queue);

  auto EventA = Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access_mode::read>(CGH);
    auto AccB = BufferB.get_access<access_mode::read>(CGH);
    auto AccC = BufferC.get_access<access_mode::write>(CGH);
    CGH.single_task<TestKernel<>>([]() {});
  });
  auto EventB = Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access_mode::read>(CGH);
    auto AccB = BufferB.get_access<access_mode::read>(CGH);
    auto AccD = BufferD.get_access<access_mode::write>(CGH);
    CGH.single_task<TestKernel<>>([]() {});
  });
  auto EventC = Queue.submit([&](handler &CGH) {
    auto AccA = BufferA.get_access<access_mode::read>(CGH);
    auto AccB = BufferB.get_access<access_mode::read>(CGH);
    auto AccE = BufferE.get_access<access_mode::write>(CGH);
    CGH.single_task<TestKernel<>>([]() {});
  });

  Graph.end_recording(Queue);

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
