//==-------- InteropKernelEnqueue.cpp --- Thread safety unit tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <gtest/gtest.h>
#include <helpers/KernelInteropCommon.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include "ThreadUtils.h"
#include "ur_mock_helpers.hpp"

namespace {
using namespace sycl;

constexpr std::size_t NArgs = 16;
constexpr std::size_t ThreadCount = 4;
constexpr std::size_t LaunchCount = 8;

uint32_t LastArgSet = -1;
std::size_t LastThread = -1;
ur_result_t redefined_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  auto Args = *params.ppArgs;
  for (uint32_t i = 0; i < *params.pnumArgs; i++) {
    if (Args[i].type != UR_EXP_KERNEL_ARG_TYPE_VALUE) {
      continue;
    }
    auto ArgIndex = Args[i].index;
    EXPECT_EQ((LastArgSet + 1) % NArgs, ArgIndex);
    LastArgSet = ArgIndex;
    std::size_t ArgValue =
        *static_cast<const std::size_t *>(Args[i].value.pointer);
    if (ArgIndex == 0)
      LastThread = ArgValue;
    else
      EXPECT_EQ(LastThread, ArgValue);
  }
  return UR_RESULT_SUCCESS;
}

TEST(KernelEnqueue, InteropKernel) {
  unittest::UrMock<> Mock;
  redefineMockForKernelInterop(Mock);
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefined_urEnqueueKernelLaunchWithArgsExp);

  platform Plt = sycl::platform();
  queue Q;

  ur_native_handle_t Handle = mock::createDummyHandle<ur_native_handle_t>();
  auto KernelCL = reinterpret_cast<typename sycl::backend_traits<
      sycl::backend::opencl>::template input_type<sycl::kernel>>(&Handle);
  auto Kernel =
      sycl::make_kernel<sycl::backend::opencl>(KernelCL, Q.get_context());

  auto TestLambda = [&](std::size_t ThreadId) {
    Q.submit([&](sycl::handler &CGH) {
       for (std::size_t I = 0; I < NArgs; ++I)
         CGH.set_arg(I, ThreadId);
       CGH.single_task(Kernel);
     }).wait();
  };

  for (std::size_t I = 0; I < LaunchCount; ++I) {
    ThreadPool Pool(ThreadCount, TestLambda);
  }
}
} // namespace
