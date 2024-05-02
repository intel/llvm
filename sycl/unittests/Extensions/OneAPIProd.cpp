//==-------- OneAPIProd.cpp --- sycl_ext_oneapi_prod unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

static bool QueueFlushed = false;

static pi_result redefinedQueueFlush(pi_queue Queue) {
  QueueFlushed = true;
  return PI_SUCCESS;
}

TEST(OneAPIProdTest, PiQueueFlush) {
  sycl::unittest::PiMock Mock(backend::ext_oneapi_level_zero);
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefine<detail::PiApiKind::piQueueFlush>(redefinedQueueFlush);
  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v};
  Queue.ext_oneapi_prod();
  EXPECT_TRUE(QueueFlushed);
  sycl::ext::oneapi::experimental::command_graph Graph(Ctx, Queue.get_device());
  Graph.begin_recording(Queue);
  try {
    Queue.ext_oneapi_prod(); // flushing while graph is recording is not allowed
    FAIL() << "Expected exception when calling ext_oneapi_prod() during graph "
              "recording not seen.";
  } catch (exception &ex) {
    EXPECT_EQ(ex.code(), make_error_code(errc::invalid));
  }
  Graph.end_recording(Queue);
}
