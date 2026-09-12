//==------- LaunchBlocking.cpp --- SYCL_LAUNCH_BLOCKING unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SYCL_LAUNCH_BLOCKING drains the queue through urQueueFinish. Counting that
// call tells whether a submission blocked, which an end-to-end test cannot do:
// there, an operation may have completed on its own by the time it returns.
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <gtest/gtest.h>
#include <helpers/CommandSubmitWrappers.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/sycl.hpp>

#include <stdexcept>

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

// A mocked adapter, a queue and a USM allocation. Construction leaves
// QueueFinishCount at zero.
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

// Only 1 enables the feature, matching CUDA_LAUNCH_BLOCKING.
TEST(LaunchBlocking, OtherValuesDoNotDrainTheQueue) {
  for (const char *Value : {"2", "-1", "10", "true", "on"}) {
    unittest::UrMock<> Mock;
    hookQueueFinish();
    unittest::ScopedEnvVar Var{LaunchBlockingName, Value, resetLaunchBlocking};

    Fixture F;
    F.Q.memset(F.Ptr, 0, 1);
    EXPECT_EQ(QueueFinishCount, 0) << "value=" << Value;
  }
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

// Both kernel spellings must block, on either queue kind.
TEST(LaunchBlocking, DrainsAfterKernel) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  for (bool Shortcut : {true, false})
    for (bool InOrder : {true, false}) {
      Fixture F{InOrder};
      unittest::single_task_wrapper<TestKernel>(Shortcut, F.Q, []() {});
      EXPECT_GE(QueueFinishCount, 1)
          << "shortcut=" << Shortcut << " in-order=" << InOrder;
    }
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

// A recording queue must not be drained: nothing executes yet and wait() is
// illegal on it.
TEST(LaunchBlocking, DoesNotDrainWhileRecording) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  ext::oneapi::experimental::command_graph Graph{F.Ctx, F.Q.get_device()};
  Graph.begin_recording(F.Q);
  unittest::single_task_wrapper<TestKernel>(/*UseShortcutFunction=*/false, F.Q,
                                            []() {});
  EXPECT_EQ(QueueFinishCount, 0);
  Graph.end_recording(F.Q);

  // Executing the finalized graph is a regular submission and does block.
  F.Q.ext_oneapi_graph(Graph.finalize());
  EXPECT_GE(QueueFinishCount, 1);
}

// Blocking mode waits from a destructor, which must stay out of the way of a
// command group that throws.
TEST(LaunchBlocking, DoesNotDrainWhenCommandGroupThrows) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  EXPECT_THROW(F.Q.submit([](handler &CGH) {
    CGH.single_task<TestKernel>([]() {});
    throw std::runtime_error("from the command group");
  }),
               std::runtime_error);
  EXPECT_EQ(QueueFinishCount, 0);

  F.Q.memset(F.Ptr, 0, 1);
  EXPECT_GE(QueueFinishCount, 1);
}

// Reports the device as supporting timestamp recording, so that a profiling tag
// takes its native path instead of falling back to a barrier.
ur_result_t redefinedDeviceGetInfoWithTimestampSupport(void *pParams) {
  auto &Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP) {
    constexpr ur_bool_t Supported = true;
    if (Params.ppPropValue)
      *static_cast<ur_bool_t *>(*Params.ppPropValue) = Supported;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(Supported);
  }
  return UR_RESULT_SUCCESS;
}

// A profiling tag is a marker as well, so it is excluded too. Its fallback path
// is a barrier, hence blocking mode would otherwise be device-dependent.
TEST(LaunchBlocking, DoesNotDrainAfterProfilingTag) {
  unittest::UrMock<> Mock;
  hookQueueFinish();
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &redefinedDeviceGetInfoWithTimestampSupport);
  unittest::ScopedEnvVar Var{LaunchBlockingName, "1", resetLaunchBlocking};

  Fixture F;
  ASSERT_TRUE(F.Q.get_device().has(aspect::ext_oneapi_queue_profiling_tag));
  F.Q.memset(F.Ptr, 0, 1);
  QueueFinishCount = 0;

  ext::oneapi::experimental::submit_profiling_tag(F.Q);
  EXPECT_EQ(QueueFinishCount, 0);
}

} // namespace
