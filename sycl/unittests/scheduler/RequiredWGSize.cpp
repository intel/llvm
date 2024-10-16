//==---- RequiredWGSize.cpp --- Check required WG size handling ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <detail/config.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

#include <helpers/TestKernel.hpp>

bool KernelGetGroupInfoCalled = false;
std::array<size_t, 3> IncomingLocalSize = {0, 0, 0};
std::array<size_t, 3> RequiredLocalSize = {0, 0, 0};

static ur_result_t redefinedKernelGetGroupInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_group_info_params_t *>(pParams);
  KernelGetGroupInfoCalled = true;
  if (*params.ppropName == UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 3 * sizeof(size_t);
    } else if (*params.ppPropValue) {
      auto size = static_cast<size_t *>(*params.ppPropValue);
      size[0] = RequiredLocalSize[0];
      size[1] = RequiredLocalSize[1];
      size[2] = RequiredLocalSize[2];
    }
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueKernelLaunch(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  if (*params.ppLocalWorkSize) {
    IncomingLocalSize[0] = (*params.ppLocalWorkSize)[0];
    IncomingLocalSize[1] = (*params.ppLocalWorkSize)[1];
    IncomingLocalSize[2] = (*params.ppLocalWorkSize)[2];
  }
  return UR_RESULT_SUCCESS;
}

static void reset() {
  KernelGetGroupInfoCalled = false;
  IncomingLocalSize = {0, 0, 0};
  RequiredLocalSize = {0, 0, 0};
}

static void performChecks() {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urEnqueueKernelLaunch",
                                           &redefinedEnqueueKernelLaunch);
  mock::getCallbacks().set_before_callback("urKernelGetGroupInfo",
                                           &redefinedKernelGetGroupInfo);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel<>>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(KernelGetGroupInfoCalled, true);
  EXPECT_EQ(IncomingLocalSize[0], RequiredLocalSize[0]);
  EXPECT_EQ(IncomingLocalSize[1], RequiredLocalSize[1]);
  EXPECT_EQ(IncomingLocalSize[2], RequiredLocalSize[2]);
}

TEST(RequiredWGSize, NoRequiredSize) {
  reset();
  performChecks();
}

TEST(RequiredWGSize, HasRequiredSize) {
  reset();
  RequiredLocalSize = {1, 2, 3};
  performChecks();
}
