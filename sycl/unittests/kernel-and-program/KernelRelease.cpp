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
#include <helpers/UrMock.hpp>
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

static ur_result_t redefinedKernelCreate(void *) {
  TestContext->KernelReferenceCount = 1;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedKernelRetain(void *) {
  ++TestContext->KernelReferenceCount;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedKernelRelease(void *) {
  --TestContext->KernelReferenceCount;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_KERNEL_INFO_CONTEXT)
      << "Unexpected kernel info requested";
  auto *Result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
  auto UrContext = detail::getSyclObjImpl(TestContext->Ctx)->getHandleRef();
  *Result = UrContext;
  return UR_RESULT_SUCCESS;
}

TEST(KernelReleaseTest, DISABLED_GetKernelRelease) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_before_callback("urKernelCreate",
                                           &redefinedKernelCreate);
  mock::getCallbacks().set_before_callback("urKernelRetain",
                                           &redefinedKernelRetain);
  mock::getCallbacks().set_before_callback("urKernelRelease",
                                           &redefinedKernelRelease);
  mock::getCallbacks().set_before_callback("urKernelGetInfo",
                                           &redefinedKernelGetInfo);

  context Ctx{sycl::platform().get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  // program Prg{Ctx};
  // Prg.build_with_source("");

  //{ kernel Krnl = Prg.get_kernel(""); }

  ASSERT_EQ(TestContext->KernelReferenceCount, 0)
      << "Reference count not equal to 0 after kernel destruction";
}
