//==-------------- KernelInfo.cpp --- kernel info unit test ----------------==//
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

using namespace sycl;

namespace {
struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  bool PrivateMemSizeCalled = false;
};
} // namespace

static std::unique_ptr<TestCtx> TestContext;

static ur_result_t redefinedKernelGetGroupInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_group_info_params_t *>(pParams);
  if (*params.ppropName == UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE) {
    TestContext->PrivateMemSizeCalled = true;
  }

  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_KERNEL_INFO_CONTEXT)
      << "Unexpected kernel info requested";
  auto *Result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
  ur_context_handle_t UrContext =
      detail::getSyclObjImpl(TestContext->Ctx)->getHandleRef();
  *Result = UrContext;
  return UR_RESULT_SUCCESS;
}

class KernelInfoTest : public ::testing::Test {
public:
  KernelInfoTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  void SetUp() override {
    mock::getCallbacks().set_before_callback("urKernelGetGroupInfo",
                                             &redefinedKernelGetGroupInfo);
    mock::getCallbacks().set_before_callback("urKernelGetInfo",
                                             &redefinedKernelGetInfo);
  }

protected:
  unittest::UrMock<> Mock;
  sycl::platform Plt;
};

TEST_F(KernelInfoTest, DISABLED_GetPrivateMemUsage) {
  context Ctx{Plt.get_devices()[0]};
  // program Prg{Ctx};
  TestContext.reset(new TestCtx(Ctx));

  // Prg.build_with_source("");

  // kernel Ker = Prg.get_kernel("");

  // Ker.get_info<info::kernel_device_specific::private_mem_size>(
  //     Ctx.get_devices()[0]);
  EXPECT_EQ(TestContext->PrivateMemSizeCalled, true)
      << "Expect piKernelGetGroupInfo to be "
      << "called with UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";
}
