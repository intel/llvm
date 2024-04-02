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
#include "detail/graph_impl.hpp"

#include <detail/config.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

using namespace sycl;
using namespace sycl::ext::oneapi;

// Common Test fixture
class CommandGraphTest : public ::testing::Test {
public:
  CommandGraphTest()
      : Mock{}, Plat{Mock.getPlatform()}, Dev{Plat.get_devices()[0]},
        Queue{Dev},
        Graph{Queue.get_context(),
              Dev,
              {experimental::property::graph::assume_buffer_outlives_graph{}}} {
  }

protected:
  void SetUp() override {}

protected:
  unittest::PiMock Mock;
  sycl::platform Plat;
  sycl::device Dev;
  sycl::queue Queue;
  experimental::command_graph<experimental::graph_state::modifiable> Graph;
};
