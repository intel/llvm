//==----------- KernelRelease.cpp --- kernel release unit test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <memory>

using namespace sycl;

namespace {
struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  int KernelReferenceCount = 0;
};
} // namespace

static std::unique_ptr<TestCtx> TestContext;

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  TestContext->KernelReferenceCount = 1;
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) {
  ++TestContext->KernelReferenceCount;
  return PI_SUCCESS;
}

static pi_result redefinedKernelRelease(pi_kernel kernel) {
  --TestContext->KernelReferenceCount;
  return PI_SUCCESS;
}

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, PI_KERNEL_INFO_CONTEXT)
      << "Unexpected kernel info requested";
  auto *Result = reinterpret_cast<RT::PiContext *>(param_value);
  RT::PiContext PiCtx =
      detail::getSyclObjImpl(TestContext->Ctx)->getHandleRef();
  *Result = PiCtx;
  return PI_SUCCESS;
}

TEST(KernelReleaseTest, DISABLED_GetKernelRelease) {
  sycl::unittest::PiMock Mock;
  Mock.redefineBefore<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefineBefore<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefineBefore<detail::PiApiKind::piKernelRelease>(
      redefinedKernelRelease);
  Mock.redefineBefore<detail::PiApiKind::piKernelGetInfo>(
      redefinedKernelGetInfo);

  context Ctx{Mock.getPlatform().get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  // program Prg{Ctx};
  // Prg.build_with_source("");

  //{ kernel Krnl = Prg.get_kernel(""); }

  ASSERT_EQ(TestContext->KernelReferenceCount, 0)
      << "Reference count not equal to 0 after kernel destruction";
}
