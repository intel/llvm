//==------------ OneAPIProd.cpp --- sycl_ext_oneapi_prod unit tests
//----------------==//
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
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefine<detail::PiApiKind::piQueueFlush>(redefinedQueueFlush);
  context Ctx{Plt};
  queue Queue{Ctx};
  Queue.ext_oneapi_prod();
  EXPECT_TRUE(QueueFlushed);
}
