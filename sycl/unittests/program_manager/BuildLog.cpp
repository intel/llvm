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
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <stdlib.h>

#include <helpers/TestKernel.hpp>

// Same as defined in config.def
static constexpr auto WarningLevelEnvVar = "SYCL_RT_WARNING_LEVEL";

static bool LogRequested = false;

static ur_result_t redefinedProgramGetBuildInfo(void *pParams) {
  auto params = *static_cast<ur_program_get_build_info_params_t *>(pParams);
  if (*params.ppPropSizeRet) {
    **params.ppPropSizeRet = 1;
  }
  if (*params.ppPropValue) {
    *static_cast<char *>(*params.ppPropValue) = '1';
  }

  if (*params.ppropName == UR_PROGRAM_BUILD_INFO_LOG) {
    LogRequested = true;
  }

  return UR_RESULT_SUCCESS;
}

static void setupCommonTestAPIs(sycl::unittest::UrMock<> &Mock) {
  using namespace sycl::detail;
  mock::getCallbacks().set_before_callback("urProgramGetBuildInfo",
                                           &redefinedProgramGetBuildInfo);
}

TEST(BuildLog, OutputNothingOnLevel1) {
  sycl::unittest::UrMock<> mock;
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "1",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
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
  sycl::unittest::UrMock<> mock;
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "2",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
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
