//==---------------- stream.cpp --- SYCL stream unit test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <limits>

#include <helpers/TestKernel.hpp>

size_t GBufferCreateCounter = 0;

static ur_result_t redefinedMemBufferCreate(void *pParams) {
  auto params = *static_cast<ur_mem_buffer_create_params_t *>(pParams);
  ++GBufferCreateCounter;
  **params.pphBuffer = nullptr;
  return UR_RESULT_SUCCESS;
}

TEST(Stream, TestStreamConstructorExceptionNoAllocation) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &redefinedMemBufferCreate);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);

  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);

    try {
      // Try to create stream with invalid workItemBufferSize parameter.
      sycl::stream InvalidStream{256, std::numeric_limits<size_t>::max(), CGH};
      FAIL() << "No exception was thrown.";
    } catch (const sycl::exception &e) {
      if (e.code() != sycl::errc::invalid)
        FAIL() << "Unexpected exception was thrown.";
    } catch (...) {
      FAIL() << "Unexpected exception was thrown.";
    }

    CGH.single_task<TestKernel<>>([=]() {});
  });

  ASSERT_EQ(GBufferCreateCounter, 0u) << "Buffers were unexpectedly created.";
}

TEST(Stream, Properties) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Queue;
  Queue
      .submit([&](sycl::handler &CGH) {
        try {
          sycl::stream Stream{256, 256, CGH, sycl::property::queue::in_order{}};
          FAIL() << "No exception was thrown.";
        } catch (const sycl::exception &e) {
          EXPECT_EQ(e.code(), sycl::errc::invalid);
          EXPECT_STREQ(e.what(),
                       "The property list contains property unsupported "
                       "for the current object");
          return;
        } catch (...) {
          FAIL() << "Unexpected exception was thrown.";
        }

        CGH.single_task<TestKernel<>>([=]() {});
      })
      .wait();
}
