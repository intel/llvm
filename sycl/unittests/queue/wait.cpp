//==----------------- wait.cpp --- queue wait unit test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace cl::sycl;

struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};

std::unique_ptr<TestCtx> TestContext;

pi_result redefinedUSMEnqueueMemset(pi_queue queue, void *ptr, pi_int32 value,
                                    size_t count,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  // Provide a dummy non-nullptr value
  *event = reinterpret_cast<pi_event>(1);
  TestContext->EventReferenceCount = 1;
  return PI_SUCCESS;
}

pi_result redefinedEventsWait(pi_uint32 num_events,
                              const pi_event *event_list) {
  ++TestContext->NEventsWaitedFor;
  return PI_SUCCESS;
}

pi_result redefinedEventGetInfo(pi_event event, pi_event_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, PI_EVENT_INFO_CONTEXT)
      << "Unexpected event info requested";
  auto *Result = reinterpret_cast<RT::PiContext *>(param_value);
  RT::PiContext PiCtx =
      detail::getSyclObjImpl(TestContext->Ctx)->getHandleRef();
  *Result = PiCtx;
  return PI_SUCCESS;
}

pi_result redefinedEventRetain(pi_event event) {
  ++TestContext->EventReferenceCount;
  return PI_SUCCESS;
}

pi_result redefinedEventRelease(pi_event event) {
  --TestContext->EventReferenceCount;
  return PI_SUCCESS;
}

// Check that the USM events are cleared from the queue upon call to wait(),
// so that they are not waited for multiple times.
TEST(QueueWaitTest, USMEventClear) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return;
  }

  // TODO: Skip test for CUDA temporarily
  if (detail::getSyclObjImpl(Plt)->getPlugin().getBackend() == backend::cuda) {
    std::cout << "Not run on CUDA - usm is not supported for CUDA backend yet"
              << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefinedUSMEnqueueMemset);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);

  context Ctx{Plt};
  TestContext.reset(new TestCtx(Ctx));
  queue Q{Ctx, default_selector()};

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Ctx);
  Q.memset(HostAlloc, 42, 1);
  Q.wait();
  ASSERT_EQ(TestContext->NEventsWaitedFor, 1);
  ASSERT_EQ(TestContext->EventReferenceCount, 0);
  Q.wait();
  ASSERT_EQ(TestContext->NEventsWaitedFor, 1);
}
