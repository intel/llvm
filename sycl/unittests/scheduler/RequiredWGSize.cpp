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
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

#include <helpers/TestKernel.hpp>

bool KernelGetGroupInfoCalled = false;
std::array<size_t, 3> IncomingLocalSize = {0, 0, 0};
std::array<size_t, 3> RequiredLocalSize = {0, 0, 0};

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                             pi_kernel_group_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  KernelGetGroupInfoCalled = true;
  if (param_name == PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    if (param_value_size_ret) {
      *param_value_size_ret = 3 * sizeof(size_t);
    } else if (param_value) {
      auto size = static_cast<size_t *>(param_value);
      size[0] = RequiredLocalSize[0];
      size[1] = RequiredLocalSize[1];
      size[2] = RequiredLocalSize[2];
    }
  }

  return PI_SUCCESS;
}

static pi_result redefinedEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                              const size_t *, const size_t *,
                                              const size_t *LocalSize,
                                              pi_uint32, const pi_event *,
                                              pi_event *) {
  if (LocalSize) {
    IncomingLocalSize[0] = LocalSize[0];
    IncomingLocalSize[1] = LocalSize[1];
    IncomingLocalSize[2] = LocalSize[2];
  }
  return PI_SUCCESS;
}

static void reset() {
  KernelGetGroupInfoCalled = false;
  IncomingLocalSize = {0, 0, 0};
  RequiredLocalSize = {0, 0, 0};
}

static void performChecks() {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);
  Mock.redefineBefore<sycl::detail::PiApiKind::piKernelGetGroupInfo>(
      redefinedKernelGetGroupInfo);

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
