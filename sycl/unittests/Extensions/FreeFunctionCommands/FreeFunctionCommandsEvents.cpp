//==------------------ FreeFunctionCommandsEvents.cpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests the behavior of khr free function commands when events can be
// discarded.

#include "FreeFunctionEventsHelpers.hpp"
#include "helpers/MockDeviceImage.hpp"
#include "helpers/MockKernelInfo.hpp"

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/khr/free_function_commands.hpp>

class TestFunctor {
public:
  void operator()() const {}
  void operator()(sycl::item<1>) const {}
  void operator()(sycl::nd_item<1> Item) const {}
};
namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestFunctor> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestFunctor"; }
  static constexpr int64_t getKernelSize() { return sizeof(TestFunctor); }
  static constexpr const char *getFileName() { return "TestFunctor.hpp"; }
  static constexpr const char *getFunctionName() {
    return "TestFunctorFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 13; }
  static constexpr unsigned getColumnNumber() { return 8; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"TestFunctor"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

namespace {

class FreeFunctionCommandsEventsTests : public ::testing::Test {
public:
  FreeFunctionCommandsEventsTests()
      : Mock{}, Queue{sycl::context(sycl::platform()), sycl::default_selector_v,
                      sycl::property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_urEnqueueKernelLaunch = 0;
    counter_urUSMEnqueueMemcpy = 0;
    counter_urUSMEnqueueFill = 0;
    counter_urUSMEnqueuePrefetch = 0;
    counter_urUSMEnqueueMemAdvise = 0;
    counter_urEnqueueEventsWaitWithBarrier = 0;
  }

  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
};

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchTaskNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_task(Handler, TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchTaskShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  sycl::khr::launch_task(Queue, TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchTaskKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);
  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_task(Handler, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchTaskShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch_task(Queue, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchForNoEvent) {

  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch(Handler, sycl::range<1>{32}, TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchForShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  sycl::khr::launch(Queue, sycl::range<1>{32}, TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchForKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch(Handler, sycl::range<1>{32}, Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchForShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch(Queue, sycl::range<1>{32}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchGroupedNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_grouped(Handler, sycl::range<1>{32}, sycl::range<1>{32},
                              TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchGroupedShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchGroupedKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);
  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_grouped(Handler, sycl::range<1>{32}, sycl::range<1>{32},
                              Kernel);
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchGroupedShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitMemcpyNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = sycl::malloc_shared<int>(N, Queue);
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::memcpy(Handler, Dst, Src, sizeof(int) * N);
  });

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Queue);
  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, MemcpyShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = sycl::malloc_shared<int>(N, Queue);
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::memcpy(Queue, Dst, Src, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Queue);
  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitCopyNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = sycl::malloc_shared<int>(N, Queue);
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::copy(Handler, Src, Dst, N);
  });

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Queue);
  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, CopyShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefined_urUSMEnqueueMemcpy);

  constexpr size_t N = 1024;
  int *Src = sycl::malloc_shared<int>(N, Queue);
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::memcpy(Queue, Dst, Src, N);

  ASSERT_EQ(counter_urUSMEnqueueMemcpy, size_t{1});

  free(Src, Queue);
  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitMemsetNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefined_urUSMEnqueueFill);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::memset(Handler, Dst, int{1}, sizeof(int) * N);
  });

  ASSERT_EQ(counter_urUSMEnqueueFill, size_t{1});

  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, MemsetShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefined_urUSMEnqueueFill);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::memset(Queue, Dst, 1, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueueFill, size_t{1});

  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitPrefetchNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            redefined_urUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::prefetch(Handler, Dst, sizeof(int) * N);
  });

  ASSERT_EQ(counter_urUSMEnqueuePrefetch, size_t{1});

  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, PrefetchShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            redefined_urUSMEnqueuePrefetch);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::prefetch(Queue, Dst, sizeof(int) * N);

  ASSERT_EQ(counter_urUSMEnqueuePrefetch, size_t{1});

  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitMemAdviseNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMAdvise",
                                            redefined_urUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::mem_advise(Handler, Dst, sizeof(int) * N, 1);
  });

  ASSERT_EQ(counter_urUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Queue);
}
TEST_F(FreeFunctionCommandsEventsTests, MemAdviseShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMAdvise",
                                            &redefined_urUSMEnqueueMemAdvise);

  constexpr size_t N = 1024;
  int *Dst = sycl::malloc_shared<int>(N, Queue);

  sycl::khr::mem_advise(Queue, Dst, sizeof(int) * N, 1);

  ASSERT_EQ(counter_urUSMEnqueueMemAdvise, size_t{1});

  free(Dst, Queue);
}

TEST_F(FreeFunctionCommandsEventsTests, BarrierBeforeHostTask) {
  // Special test for case where host_task need an event after, so a barrier is
  // enqueued to create a usable event.
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch",
                                            &redefined_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback(
      "urEnqueueEventsWaitWithBarrier", &after_urEnqueueEventsWaitWithBarrier);

  sycl::khr::launch_task(Queue, TestFunctor());

  std::chrono::time_point<std::chrono::steady_clock> HostTaskTimestamp;
  Queue
      .submit([&](sycl::handler &Handler) {
        Handler.host_task(
            [&]() { HostTaskTimestamp = std::chrono::steady_clock::now(); });
      })
      .wait();

  ASSERT_EQ(counter_urEnqueueKernelLaunch, size_t{1});
  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_urEnqueueEventsWaitWithBarrier);
}

} // namespace
