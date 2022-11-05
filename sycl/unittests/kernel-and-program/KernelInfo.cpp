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
#include <helpers/PiMock.hpp>
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

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                                             pi_kernel_group_info param_name,
                                             size_t param_value_size,
                                             void *param_value,
                                             size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE) {
    TestContext->PrivateMemSizeCalled = true;
  }

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

class KernelInfoTest : public ::testing::Test {
public:
  KernelInfoTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineBefore<detail::PiApiKind::piKernelGetGroupInfo>(
        redefinedKernelGetGroupInfo);
    Mock.redefineBefore<detail::PiApiKind::piKernelGetInfo>(
        redefinedKernelGetInfo);
  }

protected:
  unittest::PiMock Mock;
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
      << "called with PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";
}
