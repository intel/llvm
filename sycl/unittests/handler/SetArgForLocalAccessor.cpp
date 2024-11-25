//==------- SetArgForLocalAccessor.cpp --- Handler unit tests --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_mock_helpers.hpp"
#include <gtest/gtest.h>
#include <helpers/KernelInteropCommon.hpp>
#include <helpers/UrMock.hpp>

#include <sycl/sycl.hpp>

// This test checks that we pass the correct buffer size value when setting
// local_accessor as an argument through handler::set_arg to a kernel created
// using OpenCL interoperability methods.

namespace {

size_t LocalBufferArgSize = 0;

ur_result_t redefined_urKernelSetArgLocal(void *pParams) {
  auto params = *static_cast<ur_kernel_set_arg_local_params_t *>(pParams);
  LocalBufferArgSize = *params.pargSize;

  return UR_RESULT_SUCCESS;
}

TEST(HandlerSetArg, LocalAccessor) {
  sycl::unittest::UrMock<> Mock;
  redefineMockForKernelInterop(Mock);
  mock::getCallbacks().set_replace_callback("urKernelSetArgLocal",
                                            &redefined_urKernelSetArgLocal);

  constexpr size_t Size = 128;
  sycl::queue Q;

  ur_native_handle_t handle = mock::createDummyHandle<ur_native_handle_t>();
  auto KernelCL = reinterpret_cast<typename sycl::backend_traits<
      sycl::backend::opencl>::template input_type<sycl::kernel>>(&handle);
  auto Kernel =
      sycl::make_kernel<sycl::backend::opencl>(KernelCL, Q.get_context());

  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<float, 1> Acc(Size, CGH);
     CGH.set_arg(0, Acc);
     CGH.single_task(Kernel);
   }).wait();

  ASSERT_EQ(LocalBufferArgSize, Size * sizeof(float));
}
} // namespace
