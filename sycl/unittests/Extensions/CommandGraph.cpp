//==--------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/ext/oneapi/experimental/graph.hpp"
#include <sycl/sycl.hpp>

#include "detail/graph_impl.hpp"

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <gtest/gtest.h>

using namespace sycl;
using namespace sycl::ext::oneapi;

class CommandGraphTest : public ::testing::Test {
public:
  CommandGraphTest()
      : Mock{}, Plat{Mock.getPlatform()}, Dev{Plat.get_devices()[0]},
        Queue{Dev}, Graph{Queue.get_context(), Dev} {}

protected:
  void SetUp() override {}

protected:
  unittest::PiMock Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};

TEST_F(CommandGraphTest, AddNode) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl->MRoots.size() == 0);

  auto Node1 = Graph.add([&](sycl::handler &cgh) {});

  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1) != nullptr);
  ASSERT_TRUE(GraphImpl->MRoots.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 0);

  // Add a node which depends on the first
  auto Node2 = Graph.add([&](sycl::handler &cgh) {},
                         {experimental::property::node::depends_on(Node1)});
  ASSERT_TRUE(GraphImpl->MRoots.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front() ==
              sycl::detail::getSyclObjImpl(Node2));

  // Add a third node which depends on both
  auto Node3 =
      Graph.add([&](sycl::handler &cgh) {},
                {experimental::property::node::depends_on(Node1, Node2)});
  ASSERT_TRUE(GraphImpl->MRoots.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 2);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size() == 1);

  // Add a fourth node without any dependencies on the others
  auto Node4 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_TRUE(GraphImpl->MRoots.size() == 2);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 2);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MSuccessors.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node3)->MSuccessors.size() == 0);
}

TEST_F(CommandGraphTest, MakeEdge) {
  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  auto Node1 = Graph.add([&](sycl::handler &cgh) {});
  auto Node2 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 0);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size() == 0);

  Graph.make_edge(Node1, Node2);

  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size() == 1);
}

TEST_F(CommandGraphTest, BeginEndRecording) {
  sycl::queue Queue2{Dev};

  // Test throwing behaviour
  // Check we can repeatedly begin recording on the same queues
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  ASSERT_NO_THROW(Graph.begin_recording(Queue2));
  // Check we can repeatedly end recording on the same queues
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  ASSERT_NO_THROW(Graph.end_recording(Queue2));
  // Vector versions
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.begin_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));
  ASSERT_NO_THROW(Graph.end_recording({Queue, Queue2}));

  experimental::command_graph Graph2(Queue.get_context(), Dev);

  Graph.begin_recording(Queue);
  // Trying to record to a second Graph should throw
  ASSERT_ANY_THROW(Graph2.begin_recording(Queue));
  // Trying to end when it is recording to a different graph should throw
  ASSERT_ANY_THROW(Graph2.end_recording(Queue));
  Graph.end_recording(Queue);

  // Testing return values of begin and end recording
  // Queue should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue));

  // Queue2 should change state so should return true here
  ASSERT_TRUE(Graph.begin_recording(Queue2));
  // But not changed state here
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // Queue should have changed state so should return true
  ASSERT_TRUE(Graph.end_recording(Queue));
  // But not changed state here
  ASSERT_FALSE(Graph.end_recording(Queue));

  // Should end recording on Queue2
  ASSERT_TRUE(Graph.end_recording());
  // State should not change on Queue2 now
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // Testing vector begin and end
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.begin_recording(Queue));
  ASSERT_FALSE(Graph.begin_recording(Queue2));

  // End recording on both
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
  // Both shoudl now not have state changed
  ASSERT_FALSE(Graph.end_recording(Queue));
  ASSERT_FALSE(Graph.end_recording(Queue2));

  // First add one single queue
  ASSERT_TRUE(Graph.begin_recording(Queue));
  // Vector begin should still return true as Queue2 has state changed
  ASSERT_TRUE(Graph.begin_recording({Queue, Queue2}));
  // End recording on Queue2
  ASSERT_TRUE(Graph.end_recording(Queue2));
  // Vector end should still return true as Queue will have state changed
  ASSERT_TRUE(Graph.end_recording({Queue, Queue2}));
}
