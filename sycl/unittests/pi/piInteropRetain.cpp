//==--------------------- piInteropRetain.cpp -- check proper retain calls -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <detail/queue_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

namespace {
using namespace sycl;

static int QueueRetainCalled = 0;
pi_result redefinedQueueRetain(pi_queue Queue) {
  ++QueueRetainCalled;
  return PI_SUCCESS;
}

TEST(PiInteropTest, CheckRetain) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  context Ctx{Plt.get_devices()[0]};

  // The queue construction should not call to piQueueRetain. Instead
  // piQueueCreate should return the "retained" queue.
  Mock.redefineBefore<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
  queue Q{Ctx, default_selector()};
  EXPECT_TRUE(QueueRetainCalled == 0);

  cl_command_queue OCLQ = get_native<backend::opencl>(Q);
  EXPECT_TRUE(QueueRetainCalled == 1);

  // The make_queue should not call to piQueueRetain. The
  // piextCreateQueueWithNative handle should do the "retain" if needed.
  queue Q1 = make_queue<backend::opencl>(OCLQ, Ctx);
  EXPECT_TRUE(QueueRetainCalled == 1);
}

} // namespace
