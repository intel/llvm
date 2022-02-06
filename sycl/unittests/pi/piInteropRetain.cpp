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
#include <helpers/sycl_test.hpp>

namespace {
using namespace cl::sycl;

static int QueueRetainCalled = 0;
pi_result redefinedQueueRetain(pi_queue Queue) {
  ++QueueRetainCalled;
  return PI_SUCCESS;
}

static pi_result redefinedQueueGetNativeHandle(pi_queue q,
                                               pi_native_handle *nativeHandle) {
  *nativeHandle = reinterpret_cast<pi_native_handle>(q);
  return PI_SUCCESS;
}

static pi_result
redefinedQueueCreateWithNativeHandle(pi_native_handle nativeHandle, pi_context,
                                     pi_queue *queue, bool) {
  *queue = reinterpret_cast<pi_queue>(nativeHandle);
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

SYCL_TEST(PiInteropTest, CheckRetain) {
  platform Plt;
  // TODO replace with a selector
  for (const auto &Cand : platform::get_platforms()) {
    if (Cand.get_backend() == backend::opencl) {
      Plt = Cand;
      break;
    }
  }

  if (!preparePiMock(Plt))
    return;
  context Ctx{Plt.get_devices()[0]};

  // The queue construction should not call to piQueueRetain. Instead
  // piQueueCreate should return the "retained" queue.
  unittest::redefine<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
  unittest::redefine<detail::PiApiKind::piextQueueGetNativeHandle>(
      redefinedQueueGetNativeHandle);
  unittest::redefine<detail::PiApiKind::piextQueueCreateWithNativeHandle>(
      redefinedQueueCreateWithNativeHandle);
  unittest::redefine<detail::PiApiKind::piQueueGetInfo>(
      [&](pi_queue command_queue, pi_queue_info param_name,
          size_t param_value_size, void *param_value,
          size_t *param_value_size_ret) {
        if (param_name == PI_QUEUE_INFO_DEVICE) {
          if (param_value_size_ret) {
            *param_value_size_ret = sizeof(pi_device);
          }
          if (param_value) {
            device Dev = Plt.get_devices()[0];
            *static_cast<pi_device *>(param_value) =
                reinterpret_cast<pi_device>(
                    detail::getSyclObjImpl(Dev)->getNative());
          }
        }

        return PI_SUCCESS;
      });
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
