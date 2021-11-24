//==--------------------- piInteropRetain.cpp -- check proper retain calls -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>

#include <detail/queue_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

namespace {
using namespace cl::sycl;

static int QueueRetainCalled = 0;
pi_result redefinedQueueRetain(pi_queue Queue) {
  ++QueueRetainCalled;
  return PI_SUCCESS;
}

bool preparePiMock(platform &Plt) {
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return false;
  }
  if (detail::getSyclObjImpl(Plt)->getPlugin().getBackend() !=
      backend::opencl) {
    std::cout << "Not run on non-OpenCL backend" << std::endl;
    return false;
  }
  return true;
}

TEST(PiInteropTest, CheckRetain) {
  platform Plt{default_selector()};
  if (!preparePiMock(Plt))
    return;
  context Ctx{Plt.get_devices()[0]};

  unittest::PiMock Mock{Plt};

  // The queue construction should not call to piQueueRetain. Instead
  // piQueueCreate should return the "retained" queue.
  Mock.redefine<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
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
