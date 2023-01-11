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
      : Mock{}, Plat{Mock.getPlatform()}, Dev{Plat.get_devices()[0]}, Graph{} {}

protected:
  void SetUp() override {}

protected:
  unittest::PiMock Mock;
  sycl::platform Plat;
  sycl::device Dev;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};

TEST_F(CommandGraphTest, LazyQueueProperty) {
  sycl::property_list Props{
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue Queue{Dev, Props};
  bool hasProp =
      Queue.has_property<sycl::ext::oneapi::property::queue::lazy_execution>();
  ASSERT_TRUE(hasProp);
}

TEST_F(CommandGraphTest, AddNode) {
  using namespace sycl::ext::oneapi;

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  ASSERT_TRUE(GraphImpl->MRoots.size() == 0);

  auto Node1 = Graph.add([&](sycl::handler &cgh) {});

  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1) != nullptr);
  ASSERT_TRUE(GraphImpl->MRoots.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 0);

  // Add a node which depends on the first
  auto Node2 = Graph.add([&](sycl::handler &cgh) {}, {Node1});
  ASSERT_TRUE(GraphImpl->MRoots.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.front() ==
              sycl::detail::getSyclObjImpl(Node2));

  // Add a third node which depends on both
  auto Node3 = Graph.add([&](sycl::handler &cgh) {}, {Node1, Node2});
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
  using namespace sycl::ext::oneapi;

  auto GraphImpl = sycl::detail::getSyclObjImpl(Graph);

  auto Node1 = Graph.add([&](sycl::handler &cgh) {});
  auto Node2 = Graph.add([&](sycl::handler &cgh) {});
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 0);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size() == 0);

  Graph.make_edge(Node1, Node2);

  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node1)->MSuccessors.size() == 1);
  ASSERT_TRUE(sycl::detail::getSyclObjImpl(Node2)->MPredecessors.size() == 1);
}
