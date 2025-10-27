//==--------------------------- Common.hpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "sycl/ext/oneapi/experimental/graph.hpp"
#include <sycl/sycl.hpp>

#include "../../thread_safety/ThreadUtils.h"
#include "detail/graph/dynamic_impl.hpp"
#include "detail/graph/graph_impl.hpp"
#include "detail/graph/node_impl.hpp"

#include <detail/config.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

using sycl::detail::getSyclObjImpl;

// Implement the test friend class forward declared in graph_impl.hpp so tests
// can access private members to analyze internal optimizations (partitions,
// sync points).
class GraphImplTest {
  using exec_graph_impl = experimental::detail::exec_graph_impl;
  using partition = experimental::detail::partition;

public:
  static int NumPartitionsInOrder(const exec_graph_impl &Impl) {
    int NumInOrder = 0;
    for (const auto &P : Impl.MPartitions) {
      if (P && P->MIsInOrderGraph)
        ++NumInOrder;
    }
    return NumInOrder;
  }
  static int NumSyncPoints(const exec_graph_impl &Impl) {
    return Impl.MSyncPoints.size();
  }
};

// Common Test fixture
class CommandGraphTest : public ::testing::Test {
public:
  CommandGraphTest()
      : Mock{}, Plat{sycl::platform()}, Dev{Plat.get_devices()[0]}, Queue{Dev},
        Graph{Queue.get_context(),
              Dev,
              {experimental::property::graph::assume_buffer_outlives_graph{}}} {
  }

protected:
  void SetUp() override {}

protected:
  unittest::UrMock<> Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};
