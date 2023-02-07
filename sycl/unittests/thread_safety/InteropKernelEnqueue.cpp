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
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include "ThreadUtils.h"

namespace {
using namespace sycl;

constexpr std::size_t NArgs = 16;
constexpr std::size_t ThreadCount = 4;
constexpr std::size_t LaunchCount = 8;

pi_uint32 LastArgSet = -1;
std::size_t LastThread = -1;
pi_result redefined_piKernelSetArg(pi_kernel kernel, pi_uint32 arg_index,
                                   size_t arg_size, const void *arg_value) {
  EXPECT_EQ((LastArgSet + 1) % NArgs, arg_index);
  LastArgSet = arg_index;
  std::size_t ArgValue = *static_cast<const std::size_t *>(arg_value);
  if (arg_index == 0)
    LastThread = ArgValue;
  else
    EXPECT_EQ(LastThread, ArgValue);
  return PI_SUCCESS;
}

TEST(KernelEnqueue, InteropKernel) {
  unittest::PiMock Mock;
  redefineMockForKernelInterop(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piKernelSetArg>(
      redefined_piKernelSetArg);

  platform Plt = Mock.getPlatform();
  queue Q;

  DummyHandleT Handle;
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
