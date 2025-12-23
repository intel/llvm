//==------------ KernelArgs.cpp ------ Kernel arguments unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/CommandSubmitWrappers.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <condition_variable>

#include <sycl/sycl.hpp>

using namespace sycl;

class TestKernelWithIntPtr;

namespace sycl {
inline namespace _V1 {
namespace detail {

template <>
struct KernelInfo<TestKernelWithIntPtr> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const char *getName() { return "TestKernelWithIntPtr"; }
  static constexpr int64_t getKernelSize() { return sizeof(int); }

  static constexpr const detail::kernel_param_desc_t &getParamDesc(int Index) {
    return Index == 0 ? IntParamDesc : Dummy;
  }

private:
  static constexpr detail::kernel_param_desc_t IntParamDesc = {
      detail::kernel_param_kind_t::kind_std_layout, sizeof(int), 0};
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"TestKernelWithIntPtr"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

static int ArgInt = 123;

ur_result_t redefined_EnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  auto args = *params.ppArgs;
  int value = *static_cast<const int *>(args[0].value.value);

  EXPECT_EQ(value, ArgInt);

  return UR_RESULT_SUCCESS;
}

void runKernelWithArgs(queue &Queue, int ArgI) {
// Pack to 1-byte boundaries, so the kernel size is not padded
#pragma pack(push, 1)
  auto KernelLambda = [=]([[maybe_unused]] nd_item<1> i) {
    [[maybe_unused]] volatile int ArgILocal = ArgI;
  };
#pragma pack(pop)

  Queue.parallel_for<TestKernelWithIntPtr>(nd_range<1>{32, 32}, KernelLambda);
  // Erase the memory to make sure the lambda is not accessible
  std::memset(&KernelLambda, 0, sizeof(KernelLambda));
}

// This test checks, if the kernel lambda is copied properly,
// so the arguments extraction can happen after the local copy
// of the kernel lambda is deallocated.
TEST(KernelArgsTest, KernelCopy) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_EnqueueKernelLaunchWithArgsExp);

  platform Plt = sycl::platform();

  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v, property::queue::in_order()};

  std::mutex CvMutex;
  std::condition_variable Cv;
  bool ready = false;

  // The kernel submission is queued behind a host task,
  // to force the scheduler-based submission.
  Queue.submit([&](sycl::handler &CGH) {
    CGH.host_task([&] {
      std::unique_lock<std::mutex> lk(CvMutex);
      Cv.wait(lk, [&ready] { return ready; });
    });
  });

  // The kernel lambda is defined in a separate function,
  // so it will be deallocated before the argument extraction
  // and kernel submission happens.
  runKernelWithArgs(Queue, ArgInt);

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  Queue.wait();
}
