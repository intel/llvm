//==-------------- KernelInfo.cpp --- kernel info unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

using namespace sycl;

bool privateMemSizeCalled = false;

static pi_result redefinedKernelGetGroupInfo(pi_kernel kernel, pi_device device,
                               pi_kernel_group_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE) {
    privateMemSizeCalled = true;
  }

  return PI_SUCCESS;
}

class KernelInfoTest : public ::testing::Test {
public:
  KernelInfoTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
      std::clog << "This test is only supported on OpenCL devices\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    Mock->redefine<detail::PiApiKind::piKernelGetGroupInfo>(
        redefinedKernelGetGroupInfo);
  }

protected:
  platform Plt;
  std::unique_ptr<unittest::PiMock> Mock;
};

TEST_F(KernelInfoTest, GetPrivateMemUsage) {
  if (Plt.is_host()) {
    return;
  }

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_kernel_type<class GetPrivMemTest>();
  ASSERT_TRUE(Prg.has_kernel<class GetPrivMemTest>());
  kernel Ker = Prg.get_kernel<class GetPrivMemTest>();

  sycl::queue Queue{Ctx.get_devices()[0]};

  Queue.submit([](handler &CGH) { CGH.single_task<class GetPrivMemTest>([]{}); });

  Ker.get_info<info::kernel_device_specific::private_mem_size>(
      Ctx.get_devices()[0]);
  EXPECT_EQ(privateMemSizeCalled, true) << "Expect piKernelGetInfo to be "
    << "called with PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";
}
