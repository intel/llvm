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
  struct : public sycl::device_selector {
    int operator()(const sycl::device &Device) const override {
      const sycl::platform &Platform = Device.get_platform();
      if (Platform.is_host())
        return -1;
      if (Platform.get_backend() != sycl::backend::opencl &&
          Platform.get_backend() != sycl::backend::ext_oneapi_level_zero)
        return -1;
      if (!Device.has(sycl::aspect::online_compiler))
        return -1;

      return sycl::default_selector()(Device);
    }
  } DeviceSelector;

  sycl::device Dev;
  try {
    Dev = sycl::device{DeviceSelector};
  } catch (const sycl::exception &E) {
    if (E.code() == sycl::errc::runtime) {
      std::cerr << "No suitable device for the test, skipping.\n";
      return;
    }

    throw E;
  }
  sycl::unittest::PiMock Mock{Dev.get_platform()};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);

  context Ctx{Dev.get_platform()};
  queue InOrderQueue{Ctx, DeviceSelector, property::queue::in_order()};

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
