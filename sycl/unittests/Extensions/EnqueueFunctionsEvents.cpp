//==-------------------- EnqueueFunctionsEvents.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests the behavior of enqueue free functions when events can be discarded.

#include "FreeFunctionCommands/FreeFunctionEventsHelpers.hpp"

#include <helpers/TestKernel.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

using namespace sycl;

namespace oneapiext = ext::oneapi::experimental;

namespace {

class EnqueueFunctionsEventsTests : public ::testing::Test {
public:
  EnqueueFunctionsEventsTests()
      : Mock{}, Q{context(sycl::platform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_urEnqueueKernelLaunch = 0;
    counter_urUSMEnqueueMemcpy = 0;
    counter_urUSMEnqueueFill = 0;
    counter_urUSMEnqueuePrefetch = 0;
    counter_urUSMEnqueueMemAdvise = 0;
    counter_urEnqueueEventsWaitWithBarrier = 0;
  }

  unittest::UrMock<> Mock;
  queue Q;
};

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::single_task<TestKernel<>>(CGH, []() {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::single_task(CGH, Kernel); });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::single_task(Q, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for<TestKernel<>>(CGH, range<1>{32}, [](item<1>) {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::parallel_for<TestKernel<>>(Q, range<1>{32}, [](item<1>) {});

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for(CGH, range<1>{32}, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::parallel_for(Q, range<1>{32}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch<TestKernel<>>(
        CGH, nd_range<1>{range<1>{32}, range<1>{32}}, [](nd_item<1>) {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  oneapiext::nd_launch<TestKernel<>>(Q, nd_range<1>{range<1>{32}, range<1>{32}},
                                     [](nd_item<1>) {});

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch(CGH, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::nd_launch(Q, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemcpyNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::memcpy(CGH, Src, Dst, sizeof(int) * N);
  });

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemcpyShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memcpy(Q, Src, Dst, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitCopyNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::copy(CGH, Dst, Src, N); });

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, CopyShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memcpy(Q, Dst, Src, N);

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemsetNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefined_urUSMEnqueueFill);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::memset(CGH, Dst, int{1}, sizeof(int) * N);
  });

  ASSERT_EQ(counter_urUSMEnqueueFill, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemsetShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefined_urUSMEnqueueFill);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memset(Q, Dst, 1, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueueFill, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitPrefetchNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            redefined_urUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(
      Q, [&](handler &CGH) { oneapiext::prefetch(CGH, Dst, sizeof(int) * N); });

  ASSERT_EQ(counter_urUSMEnqueuePrefetch, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, PrefetchShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            redefined_urUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::prefetch(Q, Dst, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueuePrefetch, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemAdviseNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMAdvise",
                                            redefined_urUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::mem_advise(CGH, Dst, sizeof(int) * N, 1);
  });

  ASSERT_EQ(counter_urUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemAdviseShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMAdvise",
                                            &redefined_urUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::mem_advise(Q, Dst, sizeof(int) * N, 1);

  ASSERT_EQ(counter_urUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Q.submit([&](handler &CGH) {
     CGH.host_task(
         [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
   }).wait();

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_urEnqueueEventsWaitWithBarrier);
}

} // namespace
