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
  static constexpr unsigned getNumParams() { return 2; }
  static constexpr const char *getName() { return "TestKernelWithIntPtr"; }
  static constexpr int64_t getKernelSize() {
    return sizeof(int) + sizeof(void *);
  }

  static constexpr const detail::kernel_param_desc_t &getParamDesc(int Index) {
    if (Index == 0) {
      return IntParamDesc;
    } else if (Index == 1) {
      return PointerParamDesc;
    }
    return Dummy;
  }

private:
  static constexpr detail::kernel_param_desc_t IntParamDesc = {
      detail::kernel_param_kind_t::kind_std_layout, 0, 0};
  static constexpr detail::kernel_param_desc_t PointerParamDesc = {
      detail::kernel_param_kind_t::kind_pointer, 0, sizeof(int)};
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"TestKernelWithIntPtr"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

static int ArgInt = 123;

ur_result_t redefined_urKernelSetArgValue(void *pParams) {
  auto params = *static_cast<ur_kernel_set_arg_value_params_t *>(pParams);

  int ArgValue = *static_cast<const int *>(*params.ppArgValue);
  EXPECT_EQ(ArgValue, ArgInt);

  return UR_RESULT_SUCCESS;
}

static const int *ArgPointerVal = nullptr;
ur_result_t redefined_urKernelSetArgPointer(void *pParams) {
  auto params = *static_cast<ur_kernel_set_arg_pointer_params_t *>(pParams);

  ArgPointerVal = static_cast<const int *>(*params.ppArgValue);

  return UR_RESULT_SUCCESS;
}

void runKernelWithArgs(queue &Queue, int ArgI, void *ArgP) {
// Pack to 1-byte boundaries, so the kernel size is not padded
#pragma pack(push, 1)
  auto KernelLambda = [=]([[maybe_unused]] nd_item<1> i) {
    [[maybe_unused]] volatile int ArgILocal = ArgI;
    [[maybe_unused]] volatile void *ArgPLocal = ArgP;
  };
#pragma pack(pop)

  Queue.parallel_for<TestKernelWithIntPtr>(nd_range<1>{32, 32}, KernelLambda);
}

// This test checks, if the kernel lambda is copied properly,
// so the arguments extraction can happen after the local copy
// of the kernel lambda is deallocated.
TEST(KernelArgsTest, KernelCopy) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urKernelSetArgValue",
                                           &redefined_urKernelSetArgValue);
  mock::getCallbacks().set_before_callback("urKernelSetArgPointer",
                                           &redefined_urKernelSetArgPointer);

  platform Plt = sycl::platform();

  context Ctx{Plt};
  queue Queue{Ctx, default_selector_v, property::queue::in_order()};
  int *ArgPointer = (int *)sycl::malloc_device(sizeof(int), Queue);

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
  runKernelWithArgs(Queue, ArgInt, ArgPointer);

  {
    std::unique_lock<std::mutex> lk(CvMutex);
    ready = true;
  }
  Cv.notify_one();

  Queue.wait();

  ASSERT_EQ(ArgPointer, ArgPointerVal);
}
