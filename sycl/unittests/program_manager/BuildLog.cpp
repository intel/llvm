//==----------------- BuildLog.cpp --- Build log tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/internal/gtest-internal.h"
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <stdlib.h>

#include <helpers/TestKernel.hpp>

// Same as defined in config.def
static constexpr auto WarningLevelEnvVar = "SYCL_RT_WARNING_LEVEL";

static bool LogRequested = false;

static pi_result redefinedProgramGetBuildInfo(
    pi_program program, pi_device device, pi_program_build_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {

  if (param_value_size_ret) {
    *param_value_size_ret = 1;
  }
  if (param_value) {
    *static_cast<char *>(param_value) = '1';
  }

  if (param_name == PI_PROGRAM_BUILD_INFO_LOG) {
    LogRequested = true;
  }

  return PI_SUCCESS;
}

static void setupCommonTestAPIs(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefineBefore<PiApiKind::piProgramGetBuildInfo>(
      redefinedProgramGetBuildInfo);
}

TEST(BuildLog, OutputNothingOnLevel1) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "1",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  setupCommonTestAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  auto ContextImpl = getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl->getKernelProgramCache().reset();

  LogRequested = false;
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  (void)sycl::build(KernelBundle);

  EXPECT_EQ(LogRequested, false);
}

TEST(BuildLog, OutputLogOnLevel2) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "2",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  setupCommonTestAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  auto ContextImpl = getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl->getKernelProgramCache().reset();

  LogRequested = false;
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  (void)sycl::build(KernelBundle);

  EXPECT_EQ(LogRequested, true);
}
