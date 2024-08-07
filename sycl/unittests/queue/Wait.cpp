//==--------------------- Wait.cpp --- queue unit tests --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <memory>

namespace {
using namespace sycl;

struct TestCtx {
  bool SupportOOO = true;
  bool UrQueueFinishCalled = false;
  int NEventsWaitedFor = 0;
  int EventReferenceCount = 0;
};
static TestCtx TestContext;

ur_result_t redefinedQueueCreate(void *pParams) {
  auto params = *static_cast<ur_queue_create_params_t *>(pParams);
  if (!TestContext.SupportOOO &&
      (*params.ppProperties)->flags &
          UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    return UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueUSMFill(void *) {
  TestContext.EventReferenceCount = 1;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEnqueueMemBufferFill(void *) {
  TestContext.EventReferenceCount = 1;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedQueueFinish(void *) {
  TestContext.UrQueueFinishCalled = true;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventWait(void *) {
  ++TestContext.NEventsWaitedFor;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventRetain(void *) {
  ++TestContext.EventReferenceCount;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedEventRelease(void *) {
  --TestContext.EventReferenceCount;
  return UR_RESULT_SUCCESS;
}

event submitTask(queue &Q, buffer<int, 1> &Buf) {
  return Q.submit([&](handler &Cgh) {
    auto Acc = Buf.template get_access<access::mode::read_write>(Cgh);
    Cgh.fill(Acc, 42);
  });
}

TEST(QueueWait, QueueWaitTest) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urQueueCreate",
                                           &redefinedQueueCreate);
  mock::getCallbacks().set_before_callback("urQueueFinish",
                                           &redefinedQueueFinish);
  mock::getCallbacks().set_before_callback("urEnqueueUSMFill",
                                           &redefinedEnqueueUSMFill);
  mock::getCallbacks().set_before_callback("urEventWait", &redefinedEventWait);
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferFill",
                                           &redefinedEnqueueMemBufferFill);
  mock::getCallbacks().set_before_callback("urEventRetain",
                                           &redefinedEventRetain);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &redefinedEventRelease);
  context Ctx{Plt.get_devices()[0]};
  queue Q{Ctx, default_selector()};

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Ctx);

  // USM API event
  TestContext = {};
  Q.memset(HostAlloc, 42, 1);
  // No need to keep the event since we'll use urQueueFinish.
  ASSERT_EQ(TestContext.EventReferenceCount, 0);
  Q.wait();
  ASSERT_EQ(TestContext.NEventsWaitedFor, 0);
  ASSERT_TRUE(TestContext.UrQueueFinishCalled);

  // Events with temporary ownership
  {
    TestContext = {};
    buffer<int, 1> Buf{range<1>(1)};
    submitTask(Q, Buf);
    Q.wait();
    // Still owned by the execution graph
    ASSERT_EQ(TestContext.EventReferenceCount, 1);
    ASSERT_EQ(TestContext.NEventsWaitedFor, 0);
    ASSERT_TRUE(TestContext.UrQueueFinishCalled);
  }

  // Blocked commands
  {
    TestContext = {};
    buffer<int, 1> Buf{range<1>(1)};

    event DepEvent = submitTask(Q, Buf);

    // Manually block the next commands.
    std::shared_ptr<detail::event_impl> DepEventImpl =
        detail::getSyclObjImpl(DepEvent);
    auto *Cmd = static_cast<detail::Command *>(DepEventImpl->getCommand());
    Cmd->MIsBlockable = true;
    Cmd->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;

    submitTask(Q, Buf);
    submitTask(Q, Buf);

    Cmd->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueSuccess;
    Q.wait();
    // Only a single event (the last one) should be waited for here.
    ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
    ASSERT_TRUE(TestContext.UrQueueFinishCalled);
  }

  // Test behaviour for emulating an OOO queue with multiple in-order ones.
  TestContext = {};
  TestContext.SupportOOO = false;
  Q = queue{Ctx, default_selector()};
  Q.memset(HostAlloc, 42, 1);
  // The event is kept alive in this case to call wait.
  ASSERT_EQ(TestContext.EventReferenceCount, 1);
  Q.wait();
  ASSERT_EQ(TestContext.EventReferenceCount, 0);
  ASSERT_EQ(TestContext.NEventsWaitedFor, 1);
  ASSERT_FALSE(TestContext.UrQueueFinishCalled);
}

} // namespace
