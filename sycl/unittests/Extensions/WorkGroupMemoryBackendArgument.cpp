//==--------------------- WorkGroupMemoryBackendArgument.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

// Check that the work group memory object is mapped to exactly one backend
// kernel argument.
class WorkGroupMemoryKernel;
namespace syclext = sycl::ext::oneapi::experimental;
using arg_type = syclext::work_group_memory<int>;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <> struct KernelInfo<WorkGroupMemoryKernel> {
  static constexpr unsigned getNumParams() { return 1; }
  static constexpr const detail::kernel_param_desc_t &getParamDesc(int) {
    return WorkGroupMemory;
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return sizeof(arg_type); }
  static constexpr const char *getName() { return "WorkGroupMemoryKernel"; }

private:
  static constexpr detail::kernel_param_desc_t WorkGroupMemory = {
      detail::kernel_param_kind_t::kind_work_group_memory, 0, 0};
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"WorkGroupMemoryKernel"});
static sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

static int LocalMemArgs = 0;
inline ur_result_t redefined_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  auto Args = *params.ppArgs;
  for (uint32_t i = 0; i < *params.pnumArgs; i++) {
    if (Args[i].type == UR_EXP_KERNEL_ARG_TYPE_LOCAL) {
      ++LocalMemArgs;
    }
  }
  return UR_RESULT_SUCCESS;
}

TEST(URArgumentTest, URArgumentTest) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);
  sycl::platform Platform = sycl::platform();
  const sycl::device dev = Platform.get_devices()[0];
  sycl::queue q{dev};
  syclext::submit(q, [&](sycl::handler &cgh) {
    arg_type data{cgh};
    const auto kernel = [=](sycl::nd_item<1> it) { data = 42; };
    syclext::nd_launch<WorkGroupMemoryKernel>(cgh, sycl::nd_range<1>{1, 1},
                                              kernel);
  });
  q.wait();
  ASSERT_EQ(LocalMemArgs, 1);
}
