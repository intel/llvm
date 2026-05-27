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

#include <vector>

using namespace sycl;
using namespace FreeFunctionEventsHelpers;

namespace oneapiext = ext::oneapi::experimental;

namespace {

struct PrefetchCallRecord {
  const void *Ptr;
  size_t Size;
};

struct MemAdviseCallRecord {
  const void *Ptr;
  size_t Size;
  ur_usm_advice_flags_t Advice;
};

static std::vector<PrefetchCallRecord> PrefetchCallRecords;
static std::vector<MemAdviseCallRecord> MemAdviseCallRecords;
static size_t counter_urEnqueueEventsWaitWithBarrierExt = 0;

inline ur_result_t after_urUSMEnqueuePrefetchRecord(void *pParams) {
  auto Params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  PrefetchCallRecords.push_back({*Params.ppMem, *Params.psize});
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urUSMEnqueueMemAdviseRecord(void *pParams) {
  auto Params = *static_cast<ur_enqueue_usm_advise_params_t *>(pParams);
  MemAdviseCallRecords.push_back(
      {*Params.ppMem, *Params.psize, *Params.padvice});
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urEnqueueEventsWaitWithBarrierExtRecord(void *pParams) {
  (void)pParams;
  ++counter_urEnqueueEventsWaitWithBarrierExt;
  return UR_RESULT_SUCCESS;
}

class EnqueueFunctionsEventsTests : public ::testing::Test {
public:
  EnqueueFunctionsEventsTests()
      : Mock{}, Q{context(sycl::platform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_urEnqueueKernelLaunchWithArgsExp = 0;
    counter_urUSMEnqueueMemcpy = 0;
    counter_urUSMEnqueueFill = 0;
    counter_urUSMEnqueuePrefetch = 0;
    counter_urUSMEnqueueMemAdvise = 0;
    counter_urEnqueueEventsWaitWithBarrier = 0;
    counter_urEnqueueEventsWaitWithBarrierExt = 0;
    PrefetchCallRecords.clear();
    MemAdviseCallRecords.clear();
  }

  unittest::UrMock<> Mock;
  queue Q;
};

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::single_task<TestKernel>(CGH, []() {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::single_task<TestKernel>(Q, []() {});

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::single_task(CGH, Kernel); });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::single_task(Q, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for<TestKernel>(CGH, range<1>{32}, [](item<1>) {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::parallel_for<TestKernel>(Q, range<1>{32}, [](item<1>) {});

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for(CGH, range<1>{32}, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::parallel_for(Q, range<1>{32}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch<TestKernel>(
        CGH, nd_range<1>{range<1>{32}, range<1>{32}}, [](nd_item<1>) {});
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  oneapiext::nd_launch<TestKernel>(Q, nd_range<1>{range<1>{32}, range<1>{32}},
                                   [](nd_item<1>) {});

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch(CGH, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = get_kernel_id<TestKernel>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::nd_launch(Q, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
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

TEST_F(EnqueueFunctionsEventsTests, PrefetchAllFormsUseExpectedUrCalls) {
  mock::getCallbacks().set_after_callback("urEnqueueUSMPrefetch",
                                          &after_urUSMEnqueuePrefetchRecord);

  constexpr size_t N = 1024;
  constexpr size_t ChunkSize = N / 3;
  int *Memory = malloc_shared<int>(N, Q);

  oneapiext::prefetch(Q, Memory, ChunkSize);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::prefetch(CGH, Memory + ChunkSize, ChunkSize);
  });

  event E = oneapiext::submit_with_event(Q, [&](handler &CGH) {
    oneapiext::prefetch(CGH, Memory + ChunkSize * 2, ChunkSize);
  });

  E.wait();
  Q.wait();

  ASSERT_EQ(PrefetchCallRecords.size(), size_t{3});
  EXPECT_EQ(PrefetchCallRecords[0].Ptr,
            reinterpret_cast<const void *>(Memory));
  EXPECT_EQ(PrefetchCallRecords[0].Size, ChunkSize);
  EXPECT_EQ(PrefetchCallRecords[1].Ptr,
            reinterpret_cast<const void *>(Memory + ChunkSize));
  EXPECT_EQ(PrefetchCallRecords[1].Size, ChunkSize);
  EXPECT_EQ(PrefetchCallRecords[2].Ptr,
            reinterpret_cast<const void *>(Memory + ChunkSize * 2));
  EXPECT_EQ(PrefetchCallRecords[2].Size, ChunkSize);

  free(Memory, Q);
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

TEST_F(EnqueueFunctionsEventsTests, MemAdviseAllFormsUseExpectedUrCalls) {
  mock::getCallbacks().set_after_callback("urEnqueueUSMAdvise",
                                          &after_urUSMEnqueueMemAdviseRecord);

  constexpr size_t N = 1024;
  constexpr size_t ChunkSize = N / 3;
  int *Memory = malloc_shared<int>(N, Q);

  oneapiext::mem_advise(Q, Memory, ChunkSize, 0);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::mem_advise(CGH, Memory + ChunkSize, ChunkSize, 0);
  });

  event E = oneapiext::submit_with_event(Q, [&](handler &CGH) {
    oneapiext::mem_advise(CGH, Memory + ChunkSize * 2, ChunkSize, 0);
  });

  E.wait();
  Q.wait();

  ASSERT_EQ(MemAdviseCallRecords.size(), size_t{3});
  EXPECT_EQ(MemAdviseCallRecords[0].Ptr,
            reinterpret_cast<const void *>(Memory));
  EXPECT_EQ(MemAdviseCallRecords[0].Size, ChunkSize);
  EXPECT_EQ(MemAdviseCallRecords[0].Advice, ur_usm_advice_flags_t{0});

  EXPECT_EQ(MemAdviseCallRecords[1].Ptr,
            reinterpret_cast<const void *>(Memory + ChunkSize));
  EXPECT_EQ(MemAdviseCallRecords[1].Size, ChunkSize);
  EXPECT_EQ(MemAdviseCallRecords[1].Advice, ur_usm_advice_flags_t{0});

  EXPECT_EQ(MemAdviseCallRecords[2].Ptr,
            reinterpret_cast<const void *>(Memory + ChunkSize * 2));
  EXPECT_EQ(MemAdviseCallRecords[2].Size, ChunkSize);
  EXPECT_EQ(MemAdviseCallRecords[2].Advice, ur_usm_advice_flags_t{0});

  free(Memory, Q);
}

TEST_F(EnqueueFunctionsEventsTests,
       BarrierAndPartialBarrierUseExpectedUrCalls) {
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &after_urEnqueueEventsWaitWithBarrierExtRecord);

  context Ctx;
  queue Q1(Ctx, default_selector_v);

  oneapiext::single_task<TestKernel>(Q1, []() {});
  oneapiext::single_task<TestKernel>(Q1, []() {});
  oneapiext::barrier(Q1);

  oneapiext::single_task<TestKernel>(Q1, []() {});
  oneapiext::single_task<TestKernel>(Q1, []() {});
  oneapiext::barrier(Q1);

  queue Q2(Ctx, default_selector_v);
  queue Q3(Ctx, default_selector_v);

  event Event1 = oneapiext::submit_with_event(Q1, [&](handler &CGH) {
    oneapiext::single_task<TestKernel>(CGH, []() {});
  });

  event Event2 = oneapiext::submit_with_event(Q2, [&](handler &CGH) {
    oneapiext::single_task<TestKernel>(CGH, []() {});
  });

  oneapiext::partial_barrier(Q3, {Event1, Event2});
  oneapiext::single_task<TestKernel>(Q3, []() {});

  event Event3 = oneapiext::submit_with_event(Q1, [&](handler &CGH) {
    oneapiext::single_task<TestKernel>(CGH, []() {});
  });

  event Event4 = oneapiext::submit_with_event(Q2, [&](handler &CGH) {
    oneapiext::single_task<TestKernel>(CGH, []() {});
  });

  oneapiext::partial_barrier(Q3, {Event3, Event4});
  oneapiext::single_task<TestKernel>(Q3, []() {});

  Q1.wait();
  Q2.wait();
  Q3.wait();

  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrierExt, size_t{4});
}

TEST_F(EnqueueFunctionsEventsTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  oneapiext::single_task<TestKernel>(Q, []() {});

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Q.submit([&](handler &CGH) {
     CGH.host_task(
         [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
   }).wait();

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_urEnqueueEventsWaitWithBarrier);
}

} // namespace
