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

using namespace FreeFunctionEventsHelpers;

class TestFunctor {
public:
  void operator()() const {}
  void operator()(sycl::item<1>) const {}
  void operator()(sycl::nd_item<1> Item) const {}
};

class TestMoveFunctor {
public:
  static int MoveCtorCalls;

  TestMoveFunctor() = default;
  TestMoveFunctor(const TestMoveFunctor &) = default;
  TestMoveFunctor(TestMoveFunctor &&) { ++MoveCtorCalls; }
  void operator()() const {}
  void operator()(sycl::item<1>) const {}
  void operator()(sycl::nd_item<1> Item) const {}
  void operator()(sycl::nd_item<3> Item) const {}
};

int TestMoveFunctor::MoveCtorCalls = 0;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<TestFunctor> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestFunctor"; }
  static constexpr int64_t getKernelSize() { return sizeof(TestFunctor); }
};

template <>
struct KernelInfo<class TestMoveFunctor> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "TestMoveFunctor"; }
  static constexpr int64_t getKernelSize() { return sizeof(TestMoveFunctor); }
  static constexpr const char *getFileName() { return "TestMoveFunctor.hpp"; }
  static constexpr const char *getFunctionName() {
    return "TestMoveFunctorFunctionName";
  }
  static constexpr unsigned getLineNumber() { return 13; }
  static constexpr unsigned getColumnNumber() { return 8; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Imgs[2] = {
    sycl::unittest::generateDefaultImage({"TestFunctor"}),
    sycl::unittest::generateDefaultImage({"TestMoveFunctor"})};
static sycl::unittest::MockDeviceImageArray<2> ImgArray{Imgs};

namespace {

class FreeFunctionCommandsEventsTests : public ::testing::Test {
public:
  FreeFunctionCommandsEventsTests()
      : Mock{}, Queue{sycl::context(sycl::platform()), sycl::default_selector_v,
                      sycl::property::queue::in_order{}} {}

protected:
  void SetUp() override {
    counter_urEnqueueKernelLaunchWithArgsExp = 0;
    counter_urEnqueueKernelLaunchWithEvent = 0;
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
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_task(Handler, TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchTaskShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  sycl::khr::launch_task(Queue, TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchTaskKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
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

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchTaskShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch_task(Queue, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchForNoEvent) {

  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch(Handler, sycl::range<1>{32}, TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchForShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  sycl::khr::launch(Queue, sycl::range<1>{32}, TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchForKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
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

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchForShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch(Queue, sycl::range<1>{32}, Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchGroupedNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
    sycl::khr::launch_grouped(Handler, sycl::range<1>{32}, sycl::range<1>{32},
                              TestFunctor());
  });

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchGroupedShortcutNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            TestFunctor());

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests,
       LaunchGroupedShortcutMoveKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  TestMoveFunctor::MoveCtorCalls = 0;
  TestMoveFunctor MoveOnly;
  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  // This kernel submission uses scheduler-bypass path, so the HostKernel
  // shouldn't be constructed.

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            std::move(MoveOnly));

  ASSERT_EQ(TestMoveFunctor::MoveCtorCalls, 0);
  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});

  // Another kernel submission is queued behind a host task,
  // to force the scheduler-based submission. In this case, the HostKernel
  // should be constructed.

  // Replace the callback with an event based one, since the scheduler
  // needs to create an event internally
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithEvent);

  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
    });
  });

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            std::move(MoveOnly));

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  Queue.wait();

  // Move ctor for TestMoveFunctor is called during move construction of
  // HostKernel. Copy ctor is called by InstantiateKernelOnHost, can't delete
  // it.
  ASSERT_EQ(TestMoveFunctor::MoveCtorCalls, 1);
  ASSERT_EQ(counter_urEnqueueKernelLaunchWithEvent, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchTaskShortcutMoveKernel) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  TestMoveFunctor::MoveCtorCalls = 0;
  TestMoveFunctor MoveOnly;
  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  // This kernel submission uses scheduler-bypass path, so the HostKernel
  // shouldn't be constructed.

  sycl::khr::launch_task(Queue, std::move(MoveOnly));

  ASSERT_EQ(TestMoveFunctor::MoveCtorCalls, 0);
  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});

  // Another kernel submission is queued behind a host task,
  // to force the scheduler-based submission. In this case, the HostKernel
  // should be constructed.

  // Replace the callback with an event based one, since the scheduler
  // needs to create an event internally
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithEvent);

  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
    });
  });

  sycl::khr::launch_task(Queue, std::move(MoveOnly));

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  Queue.wait();

  // Move ctor for TestMoveFunctor is called during move construction of
  // HostKernel. Copy ctor is called by InstantiateKernelOnHost, can't delete
  // it.
  ASSERT_EQ(TestMoveFunctor::MoveCtorCalls, 1);
  ASSERT_EQ(counter_urEnqueueKernelLaunchWithEvent, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, SubmitLaunchGroupedKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
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

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
}

TEST_F(FreeFunctionCommandsEventsTests, LaunchGroupedShortcutKernelNoEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &after_urKernelGetInfo);

  auto KID = sycl::get_kernel_id<TestFunctor>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Queue.get_context(), std::vector<sycl::kernel_id>{KID});

  ASSERT_TRUE(KB.has_kernel(KID));

  auto Kernel = KB.get_kernel(KID);

  sycl::khr::launch_grouped(Queue, sycl::range<1>{32}, sycl::range<1>{32},
                            Kernel);

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
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
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
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

  ASSERT_EQ(counter_urEnqueueKernelLaunchWithArgsExp, size_t{1});
  ASSERT_EQ(counter_urEnqueueEventsWaitWithBarrier, size_t{1});
  ASSERT_TRUE(HostTaskTimestamp > timestamp_urEnqueueEventsWaitWithBarrier);
}

} // namespace
