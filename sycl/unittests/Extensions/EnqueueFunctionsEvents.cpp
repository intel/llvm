//==-------------------- EnqueueFunctionsEvents.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests the behavior of enqueue free functions when events can be discarded.

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

namespace oneapiext = ext::oneapi::experimental;

namespace {

inline pi_result after_piKernelGetInfo(pi_kernel kernel,
                                       pi_kernel_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  constexpr char MockKernel[] = "TestKernel";
  if (param_name == PI_KERNEL_INFO_FUNCTION_NAME) {
    if (param_value) {
      assert(param_value_size == sizeof(MockKernel));
      std::memcpy(param_value, MockKernel, sizeof(MockKernel));
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(MockKernel);
  }
  return PI_SUCCESS;
}

thread_local size_t counter_piEnqueueKernelLaunch = 0;
inline pi_result redefined_piEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                                 const size_t *, const size_t *,
                                                 const size_t *, pi_uint32,
                                                 const pi_event *,
                                                 pi_event *event) {
  ++counter_piEnqueueKernelLaunch;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piextUSMEnqueueMemcpy = 0;
inline pi_result redefined_piextUSMEnqueueMemcpy(pi_queue, pi_bool, void *,
                                                 const void *, size_t,
                                                 pi_uint32, const pi_event *,
                                                 pi_event *event) {
  ++counter_piextUSMEnqueueMemcpy;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piextUSMEnqueueMemset = 0;
inline pi_result redefined_piextUSMEnqueueMemset(pi_queue, void *, pi_int32,
                                                 size_t, pi_uint32,
                                                 const pi_event *,
                                                 pi_event *event) {
  ++counter_piextUSMEnqueueMemset;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piextUSMEnqueuePrefetch = 0;
inline pi_result redefined_piextUSMEnqueuePrefetch(pi_queue, const void *,
                                                   size_t,
                                                   pi_usm_migration_flags,
                                                   pi_uint32, const pi_event *,
                                                   pi_event *event) {
  ++counter_piextUSMEnqueuePrefetch;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piextUSMEnqueueMemAdvise = 0;
inline pi_result redefined_piextUSMEnqueueMemAdvise(pi_queue, const void *,
                                                    size_t, pi_mem_advice,
                                                    pi_event *event) {
  ++counter_piextUSMEnqueueMemAdvise;
  EXPECT_EQ(event, nullptr);
  return PI_SUCCESS;
}

thread_local size_t counter_piEnqueueEventsWaitWithBarrier = 0;
thread_local std::chrono::time_point<std::chrono::steady_clock>
    timestamp_piEnqueueEventsWaitWithBarrier;
inline pi_result after_piEnqueueEventsWaitWithBarrier(pi_queue, pi_uint32,
                                                      const pi_event *,
                                                      pi_event *) {
  ++counter_piEnqueueEventsWaitWithBarrier;
  timestamp_piEnqueueEventsWaitWithBarrier = std::chrono::steady_clock::now();
  return PI_SUCCESS;
}

class EnqueueFunctionsEventsTests : public ::testing::Test {
public:
  EnqueueFunctionsEventsTests()
      : Mock{}, Q{context(Mock.getPlatform()), default_selector_v,
                  property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_piEnqueueKernelLaunch = 0;
    counter_piextUSMEnqueueMemcpy = 0;
    counter_piextUSMEnqueueMemset = 0;
    counter_piextUSMEnqueuePrefetch = 0;
    counter_piextUSMEnqueueMemAdvise = 0;
    counter_piEnqueueEventsWaitWithBarrier = 0;
  }

  unittest::PiMock Mock;
  queue Q;
};

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::single_task<TestKernel<>>(CGH, []() {});
  });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitSingleTaskKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::single_task(CGH, Kernel); });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SingleTaskShortcutKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::single_task(Q, Kernel);

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for<TestKernel<>>(CGH, range<1>{32}, [](item<1>) {});
  });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::parallel_for<TestKernel<>>(Q, range<1>{32}, [](item<1>) {});

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitRangeParallelForKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::parallel_for(CGH, range<1>{32}, Kernel);
  });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, RangeParallelForShortcutKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::parallel_for(Q, range<1>{32}, Kernel);

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch<TestKernel<>>(
        CGH, nd_range<1>{range<1>{32}, range<1>{32}}, [](nd_item<1>) {});
  });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);

  oneapiext::nd_launch<TestKernel<>>(Q, nd_range<1>{range<1>{32}, range<1>{32}},
                                     [](nd_item<1>) {});

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitNDLaunchKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::nd_launch(CGH, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);
  });

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, NDLaunchShortcutKernelNoEvent) {
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piKernelGetInfo>(after_piKernelGetInfo);

  auto KID = get_kernel_id<TestKernel<>>();
  auto KB = get_kernel_bundle<bundle_state::executable>(
      Q.get_context(), std::vector<kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  oneapiext::nd_launch(Q, nd_range<1>{range<1>{32}, range<1>{32}}, Kernel);

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemcpyNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefined_piextUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::memcpy(CGH, Src, Dst, sizeof(int) * N);
  });

  ASSERT_EQ(counter_piextUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemcpyShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefined_piextUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memcpy(Q, Src, Dst, sizeof(int) * N);

  ASSERT_EQ(counter_piextUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitCopyNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefined_piextUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q,
                    [&](handler &CGH) { oneapiext::copy(CGH, Dst, Src, N); });

  ASSERT_EQ(counter_piextUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, CopyShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefined_piextUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = malloc_shared<int>(N, Q);
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memcpy(Q, Dst, Src, N);

  ASSERT_EQ(counter_piextUSMEnqueueMemcpy, size_t{1});

  free(Src, Q);
  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemsetNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefined_piextUSMEnqueueMemset);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::memset(CGH, Dst, int{1}, sizeof(int) * N);
  });

  ASSERT_EQ(counter_piextUSMEnqueueMemset, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemsetShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefined_piextUSMEnqueueMemset);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::memset(Q, Dst, 1, sizeof(int) * N);

  ASSERT_EQ(counter_piextUSMEnqueueMemset, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitPrefetchNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueuePrefetch>(
      redefined_piextUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(
      Q, [&](handler &CGH) { oneapiext::prefetch(CGH, Dst, sizeof(int) * N); });

  ASSERT_EQ(counter_piextUSMEnqueuePrefetch, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, PrefetchShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueuePrefetch>(
      redefined_piextUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::prefetch(Q, Dst, sizeof(int) * N);

  ASSERT_EQ(counter_piextUSMEnqueuePrefetch, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, SubmitMemAdviseNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemAdvise>(
      redefined_piextUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::submit(Q, [&](handler &CGH) {
    oneapiext::mem_advise(CGH, Dst, sizeof(int) * N, 1);
  });

  ASSERT_EQ(counter_piextUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, MemAdviseShortcutNoEvent) {
  Mock.redefine<detail::PiApiKind::piextUSMEnqueueMemAdvise>(
      redefined_piextUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = malloc_shared<int>(N, Q);

  oneapiext::mem_advise(Q, Dst, sizeof(int) * N, 1);

  ASSERT_EQ(counter_piextUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Q);
}

TEST_F(EnqueueFunctionsEventsTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefined_piEnqueueKernelLaunch);
  Mock.redefineAfter<detail::PiApiKind::piEnqueueEventsWaitWithBarrier>(
      after_piEnqueueEventsWaitWithBarrier);

  oneapiext::single_task<TestKernel<>>(Q, []() {});

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Q.submit([&](handler &CGH) {
     CGH.host_task(
         [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
   }).wait();

  ASSERT_EQ(counter_piEnqueueKernelLaunch, size_t{1});
  ASSERT_EQ(counter_piEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_piEnqueueEventsWaitWithBarrier);
}

} // namespace
