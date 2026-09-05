//==------- LaunchBlocking.cpp --- SYCL_LAUNCH_BLOCKING unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SYCL_LAUNCH_BLOCKING makes every submission synchronous by draining the
// queue, which the runtime does through urQueueFinish. Counting that call is
// what lets these tests observe the two things an end-to-end test cannot:
// that nothing is drained when the variable is unset or zero, and that barriers
// stay excluded even when it is set.
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <gtest/gtest.h>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

namespace {
using namespace sycl;

int QueueFinishCount = 0;

ur_result_t redefinedQueueFinish(void *) {
  ++QueueFinishCount;
  return UR_RESULT_SUCCESS;
}

const char *LaunchBlockingName =
    detail::SYCLConfig<detail::SYCL_LAUNCH_BLOCKING>::getName();

auto resetLaunchBlocking =
    detail::SYCLConfig<detail::SYCL_LAUNCH_BLOCKING>::reset;

// Everything a test needs: a mocked adapter, a queue, and a USM allocation to
// operate on. Construction leaves QueueFinishCount at zero.
struct Fixture {
  Fixture(bool InOrder = true)
      : Plt{platform()}, Ctx{Plt.get_devices()[0]},
        Q{InOrder ? queue{Ctx, default_selector_v, property::queue::in_order{}}
                  : queue{Ctx, default_selector_v}},
        Ptr{malloc_host(1, Ctx)} {
    QueueFinishCount = 0;
  }

  ~Fixture() { free(Ptr, Ctx); }

  platform Plt;
  context Ctx;
  queue Q;
  void *Ptr;
};

// Registers the urQueueFinish hook. Must be called after UrMock is constructed.
void hookQueueFinish() {
  mock::getCallbacks().set_before_callback("urQueueFinish",
                                           &redefinedQueueFinish);
}

TEST(LaunchBlocking, UnsetDoesNotDrainTheQueue) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, nullptr, resetLaunchBlocking};

  Fixture F;
  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_EQ(QueueFinishCount, 0);
}

TEST(LaunchBlocking, ZeroDoesNotDrainTheQueue) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "0", resetLaunchBlocking};

  Fixture F;
  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_EQ(QueueFinishCount, 0);
}

TEST(LaunchBlocking, DrainsAfterMemoryOperation) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_GE(QueueFinishCount, 1);
}

// Any non-zero value enables the feature, matching CUDA_LAUNCH_BLOCKING.
TEST(LaunchBlocking, NonZeroValueDrainsTheQueue) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "2", resetLaunchBlocking};

  Fixture F;
  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_GE(QueueFinishCount, 1);
}

// A command group goes through submit_impl rather than the memory operation
// fast path.
TEST(LaunchBlocking, DrainsAfterHandlerSubmission) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  void *Ptr = F.Ptr;
  F.Q.submit([&](handler &CGH) { CGH.memset(Ptr, 0, 1); });
  EXPECT_GE(QueueFinishCount, 1);
}

TEST(LaunchBlocking, DrainsOnOutOfOrderQueue) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F{/*InOrder=*/false};
  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_GE(QueueFinishCount, 1);
}

// Barriers are excluded from blocking mode. This is the case no end-to-end test
// covers: an event handed to a barrier always has its work already enqueued, so
// the exclusion never changes an observable outcome there.
TEST(LaunchBlocking, DoesNotDrainAfterHandlerBarrier) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  void *Ptr = F.Ptr;
  // Put something on the queue first so the barrier is not trivially empty,
  // then ignore the drain that submission itself performs.
  F.Q.memset(Ptr, 0, 1);
  QueueFinishCount = 0;

  F.Q.submit([&](handler &CGH) { CGH.ext_oneapi_barrier(); });
  EXPECT_EQ(QueueFinishCount, 0);
}

TEST(LaunchBlocking, DoesNotDrainAfterHandlerBarrierWithWaitList) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  void *Ptr = F.Ptr;
  event E = F.Q.memset(Ptr, 0, 1);
  QueueFinishCount = 0;

  F.Q.submit([&](handler &CGH) { CGH.ext_oneapi_barrier({E}); });
  EXPECT_EQ(QueueFinishCount, 0);
}

TEST(LaunchBlocking, DoesNotDrainAfterQueueBarrier) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  F.Q.memset(F.Ptr, 0, 1);
  QueueFinishCount = 0;

  F.Q.ext_oneapi_submit_barrier();
  EXPECT_EQ(QueueFinishCount, 0);
}

} // namespace
