//==----------- KernelRelease.cpp --- kernel release unit test -------------==//
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

#include <iostream>
#include <memory>

using namespace cl::sycl;

namespace {
struct TestCtx {
  TestCtx(context &Ctx) : Ctx{Ctx} {};

  context &Ctx;
  int KernelReferenceCount = 0;
};
} // namespace

static std::unique_ptr<TestCtx> TestContext;

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

static pi_result redefinedKernelSetExecInfo(pi_kernel kernel,
                                            pi_kernel_exec_info param_name,
                                            size_t param_value_size,
                                            const void *param_value) {
  return PI_SUCCESS;
}

TEST(KernelReleaseTest, GetKernelRelease) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "The program/kernel methods are mostly no-op on the host "
                 "device, the test is not run."
              << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);
  Mock.redefine<detail::PiApiKind::piKernelSetExecInfo>(
      redefinedKernelSetExecInfo);

  context Ctx{Plt.get_devices()[0]};
  TestContext.reset(new TestCtx(Ctx));

  program Prg{Ctx};
  Prg.build_with_source("");

  { kernel Krnl = Prg.get_kernel(""); }

  ASSERT_EQ(TestContext->KernelReferenceCount, 0)
      << "Reference count not equal to 0 after kernel destruction";
}
