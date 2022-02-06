//==----------------- BuildLog.cpp --- Build log tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/internal/gtest-internal.h"
#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/sycl_test.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <stdlib.h>

#include <helpers/TestKernel.hpp>

// Same as defined in config.def
static constexpr auto WarningLevelEnvVar = "SYCL_RT_WARNING_LEVEL";

static bool LogRequested = false;

static pi_result redefinedProgramGetBuildInfo(
    pi_program program, pi_device device, cl_program_build_info param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {

  if (param_name == PI_PROGRAM_BUILD_INFO_LOG) {
    LogRequested = true;
    std::string Log = "Some log";
    if (param_value_size_ret) {
      *param_value_size_ret = Log.size() + 1;
    }
    if (param_value) {
      strncpy(static_cast<char *>(param_value), Log.data(), Log.size() + 1);
    }
  }

  return PI_SUCCESS;
}

static pi_result redefinedProgramGetInfo(pi_program program,
                                         pi_program_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
  if (param_name == PI_PROGRAM_INFO_DEVICES) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(size_t);
    }
    if (param_value) {
      *static_cast<size_t *>(param_value) = 1;
    }
  }

  return PI_SUCCESS;
}

static pi_result redefinedDeviceGetInfo(pi_device device,
                                        pi_device_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_NAME) {
    const std::string name = "Test Device";
    if (param_value_size_ret) {
      *param_value_size_ret = name.size();
    }
    if (param_value) {
      auto *val = static_cast<char *>(param_value);
      strcpy(val, name.data());
    }
  }
  if (param_name == PI_DEVICE_INFO_COMPILER_AVAILABLE) {
    if (param_value_size_ret) {
      *param_value_size_ret = sizeof(cl_bool);
    }
    if (param_value) {
      auto *val = static_cast<cl_bool *>(param_value);
      *val = 1;
    }
  }
  return PI_SUCCESS;
}

static void setupCommonTestAPIs() {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  redefine<PiApiKind::piProgramGetBuildInfo>(redefinedProgramGetBuildInfo);
  redefine<PiApiKind::piProgramGetInfo>(redefinedProgramGetInfo);
  redefine<PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
}

SYCL_TEST(BuildLog, OutputNothingOnLevel1) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "1",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::platform Plt;
  for (auto &Cur : sycl::platform::get_platforms()) {
    if (Cur.get_backend() == sycl::backend::opencl) {
      Plt = Cur;
      break;
    }
  }

  setupCommonTestAPIs();

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  sycl::context Ctx = Queue.get_context();
  auto ContextImpl = getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl->getKernelProgramCache().reset();

  LogRequested = false;
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  (void)sycl::build(KernelBundle);

  EXPECT_EQ(LogRequested, false);
}

SYCL_TEST(BuildLog, OutputLogOnLevel2) {
  using namespace sycl::detail;
  using namespace sycl::unittest;
  ScopedEnvVar var(WarningLevelEnvVar, "2",
                   SYCLConfig<SYCL_RT_WARNING_LEVEL>::reset);

  sycl::platform Plt;
  for (auto &Cur : sycl::platform::get_platforms()) {
    if (Cur.get_backend() == sycl::backend::opencl) {
      Plt = Cur;
      break;
    }
  }

  setupCommonTestAPIs();

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  auto ContextImpl = getSyclObjImpl(Ctx);
  // Make sure no kernels are cached
  ContextImpl->getKernelProgramCache().reset();

  LogRequested = false;
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  (void)sycl::build(KernelBundle);

  EXPECT_EQ(LogRequested, true);
}
