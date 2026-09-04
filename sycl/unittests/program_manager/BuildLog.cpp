//==----------------- BuildLog.cpp --- Build log tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gtest/internal/gtest-internal.h"

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
  // Mimic the UR/OpenCL contract for string build-info queries: the reported
  // size includes the null terminator and the written buffer is null
  // terminated. The runtime (ProgramManager::getProgramBuildLog) relies on this
  // when constructing a std::string from the raw pointer; reporting a size of 1
  // and writing a single non-null byte makes it strlen past the allocation
  // (heap-buffer-overflow). sizeof("1") == 2 covers the '1' and the '\0'.
  static constexpr char Log[] = "1";
  if (*params.ppPropSizeRet) {
    **params.ppPropSizeRet = sizeof(Log);
  }
  if (*params.ppPropValue) {
    std::memcpy(*params.ppPropValue, Log, sizeof(Log));
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

  context_impl &ContextImpl = *getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl.getKernelProgramCache().reset();

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

  context_impl &ContextImpl = *getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl.getKernelProgramCache().reset();

  LogRequested = false;
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  (void)sycl::build(KernelBundle);

  EXPECT_EQ(LogRequested, true);
}
