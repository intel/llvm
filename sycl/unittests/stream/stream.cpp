//==---------------- stream.cpp --- SYCL stream unit test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <limits>

#include <helpers/TestKernel.hpp>

size_t GBufferCreateCounter = 0;

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  ++GBufferCreateCounter;
  *ret_mem = nullptr;
  return PI_SUCCESS;
}

TEST(Stream, TestStreamConstructorExceptionNoAllocation) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piMemBufferCreate>(
      redefinedMemBufferCreate);

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
