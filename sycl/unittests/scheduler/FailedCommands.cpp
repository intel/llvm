//==----------- FailedCommands.cpp ---- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

using namespace sycl;

TEST_F(SchedulerTest, FailedDependency) {
  unittest::UrMock<> Mock;
  platform Plt = sycl::platform();
  queue Queue(context(Plt), default_selector_v);
  sycl::detail::queue_impl &QueueImpl = *detail::getSyclObjImpl(Queue);

  detail::Requirement MockReq = getMockRequirement();
  MockCommand MDep(&QueueImpl);
  MockCommand MUser(&QueueImpl);
  MDep.addUser(&MUser);
  std::vector<detail::Command *> ToCleanUp;
  (void)MUser.addDep(detail::DepDesc{&MDep, &MockReq, nullptr}, ToCleanUp);
  MUser.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  MDep.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueFailed;

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued =
      MockScheduler::enqueueCommand(&MUser, Res, detail::NON_BLOCKING);

  ASSERT_FALSE(Enqueued) << "Enqueue process must fail\n";
  ASSERT_EQ(Res.MCmd, &MDep) << "Wrong failed command\n";
  ASSERT_EQ(Res.MResult, detail::EnqueueResultT::SyclEnqueueFailed)
      << "Enqueue process must fail\n";
  ASSERT_EQ(MUser.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueReady)
      << "MUser shouldn't be marked as failed\n";
  ASSERT_EQ(MDep.MEnqueueStatus, detail::EnqueueResultT::SyclEnqueueFailed)
      << "MDep should be marked as failed\n";
}

void RunWithFailedCommandsAndCheck(bool SyncExceptionExpected,
                                   int AsyncExceptionCountExpected) {
  platform Plt = sycl::platform();
  int ExceptionListSize = 0;
  sycl::async_handler AsyncHandler =
      [&ExceptionListSize](sycl::exception_list ExceptionList) {
        ExceptionListSize = ExceptionList.size();
      };
  bool ExceptionThrown = false;
  queue Queue(context(Plt), default_selector_v, AsyncHandler);
  {
    int initVal = 0;
    sycl::buffer<int, 1> Buf(&initVal, 1);
    try {
      Queue.submit([&](sycl::handler &CGH) {
        Buf.get_access<sycl::access::mode::write>(CGH);
        CGH.single_task<TestKernel>([]() {});
      });
    } catch (...) {
      ExceptionThrown = true;
    }
  }
  EXPECT_EQ(ExceptionThrown, SyncExceptionExpected);
  Queue.wait_and_throw();
  EXPECT_EQ(ExceptionListSize, AsyncExceptionCountExpected);
}

ur_result_t failingUrCall(void *) { return UR_RESULT_ERROR_UNKNOWN; }

TEST_F(SchedulerTest, FailedKernelException) {
  unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunchWithArgsExp",
                                           &failingUrCall);
  RunWithFailedCommandsAndCheck(true, 0);
}

TEST_F(SchedulerTest, FailedCopyBackException) {
  unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueMemBufferRead",
                                           &failingUrCall);
  RunWithFailedCommandsAndCheck(false, 1);
}

bool DummyEventReturned = false;
bool DummyEventReleaseAttempt = false;
ur_event_handle_t DummyEvent = mock::createDummyHandle<ur_event_handle_t>();

inline ur_result_t failedEnqueueKernelLaunchWithDummy(void *pParams) {
  DummyEventReturned = true;
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  **params.pphEvent = DummyEvent;
  return UR_RESULT_ERROR_UNKNOWN;
}

inline ur_result_t checkDummyInEventRelease(void *pParams) {
  auto params = static_cast<ur_event_handle_t>(pParams);
  DummyEventReleaseAttempt = params == DummyEvent;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t failedEnqueueBarrierWithDummy(void *pParams) {
  DummyEventReturned = true;
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  **params.pphEvent = DummyEvent;
  return UR_RESULT_ERROR_UNKNOWN;
}

// Checks that in case of failed command and "valid" event assigned to output
// event var, RT ignores it and do not call release since its usage is undefined
// behavior.
TEST(FailedCommandsTest, CheckUREventReleaseWithKernel) {
  DummyEventReleaseAttempt = false;
  DummyEventReturned = false;
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunchWithArgsExp",
                                           &failedEnqueueKernelLaunchWithDummy);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &checkDummyInEventRelease);
  platform Plt = sycl::platform();
  queue Queue(context(Plt), default_selector_v);
  {
    try {
      Queue.submit(
          [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
    } catch (...) {
    }
  }
  Queue.wait();
  ASSERT_TRUE(DummyEventReturned);
  ASSERT_FALSE(DummyEventReleaseAttempt);
}

// Checks that in case of failed command and "valid" event assigned to output
// event var, RT ignores it and do not call release since its usage is undefined
// behavior.
TEST(FailedCommandsTest, CheckUREventReleaseWithBarrier) {
  DummyEventReleaseAttempt = false;
  DummyEventReturned = false;
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urEnqueueEventsWaitWithBarrierExt",
                                           &failedEnqueueBarrierWithDummy);
  mock::getCallbacks().set_before_callback("urEventRelease",
                                           &checkDummyInEventRelease);
  platform Plt = sycl::platform();
  queue Queue(context(Plt), default_selector_v);
  {
    try {
      Queue.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); });
    } catch (...) {
    }
  }
  Queue.wait();
  ASSERT_TRUE(DummyEventReturned);
  ASSERT_FALSE(DummyEventReleaseAttempt);
}
