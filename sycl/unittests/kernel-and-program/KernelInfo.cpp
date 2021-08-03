//==-------------- KernelInfo.cpp --- kernel info unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

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

static pi_result redefinedProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  return PI_SUCCESS;
}

static pi_result
redefinedProgramBuild(pi_program program, pi_uint32 num_devices,
                      const pi_device *device_list, const char *options,
                      void (*pfn_notify)(pi_program program, void *user_data),
                      void *user_data) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelRelease(pi_kernel kernel) { return PI_SUCCESS; }

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

static pi_result redefinedKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info param_name,
                                            size_t param_value_size,
                                            const void *param_value) {
  return PI_SUCCESS;
}

class KernelInfoTest : public ::testing::Test {
public:
  KernelInfoTest() : Plt{default_selector()} {}

protected:
  void SetUp() override {
    if (Plt.is_host()) {
      std::clog << "This test is only supported on non-host platforms.\n";
      std::clog << "Current platform is "
                << Plt.get_info<info::platform::name>();
      return;
    }

    Mock = std::make_unique<unittest::PiMock>(Plt);

    Mock->redefine<detail::PiApiKind::piKernelGetGroupInfo>(
        redefinedKernelGetGroupInfo);
    Mock->redefine<detail::PiApiKind::piclProgramCreateWithSource>(
        redefinedProgramCreateWithSource);
    Mock->redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
    Mock->redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
    Mock->redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
    Mock->redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
    Mock->redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
    Mock->redefine<detail::PiApiKind::piKernelSetExecInfo>(
        redefinedKernelSetExecInfo);
  }

protected:
  platform Plt;
  std::unique_ptr<unittest::PiMock> Mock;
};

TEST_F(KernelInfoTest, GetPrivateMemUsage) {
  if (Plt.is_host()) {
    return;
  }

  context Ctx{Plt.get_devices()[0]};
  program Prg{Ctx};
  TestContext.reset(new TestCtx(Ctx));

  Prg.build_with_source("");

  kernel Ker = Prg.get_kernel("");

  Ker.get_info<info::kernel_device_specific::private_mem_size>(
      Ctx.get_devices()[0]);
  EXPECT_EQ(TestContext->PrivateMemSizeCalled, true)
      << "Expect piKernelGetGroupInfo to be "
      << "called with PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";

  TestContext->PrivateMemSizeCalled = false;
  Ker.get_work_group_info<info::kernel_work_group::private_mem_size>(
      Ctx.get_devices()[0]);
  EXPECT_EQ(TestContext->PrivateMemSizeCalled, true)
      << "Expect piKernelGetGroupInfo to be "
      << "called with PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE";
}
