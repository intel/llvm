//==-------- InOrderQueueHostTaskDeps.cpp --- Scheduler unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

#include <helpers/TestKernel.hpp>

using namespace sycl;

size_t GEventsWaitCounter = 0;

inline pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  if (num_events > 0) {
    GEventsWaitCounter++;
  }
  return PI_SUCCESS;
}

TEST_F(SchedulerTest, InOrderQueueHostTaskDeps) {
  default_selector Selector;
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  // This test only contains device image for SPIR-V capable devices.
  if (Plt.get_backend() != sycl::backend::opencl &&
      Plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Only OpenCL and Level Zero are supported for this test\n";
    return;
  }

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);

  context Ctx{Plt};
  queue InOrderQueue{Ctx, Selector, property::queue::in_order()};

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  event Evt = InOrderQueue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel>([] {});
  });
  InOrderQueue
      .submit([&](sycl::handler &CGH) {
        CGH.use_kernel_bundle(ExecBundle);
        CGH.host_task([=] {});
      })
      .wait();

  EXPECT_TRUE(GEventsWaitCounter == 1);
}
