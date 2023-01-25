//==---- itt_annotations.cpp --- ITT Annotations for SPIR-V images ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <detail/config.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

#include <helpers/TestKernel.hpp>

// Same as defined in config.def
static constexpr auto ITTProfileEnvVarName = "INTEL_ENABLE_OFFLOAD_ANNOTATIONS";

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
  (void)_putenv_s(name, value);
#else
  (void)setenv(name, value, /*overwrite*/ 1);
#endif
}

static void unset_env(const char *name) {
#ifdef _WIN32
  (void)_putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

bool HasITTEnabled = false;

static pi_result
redefinedProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                          size_t spec_size,
                                          const void *spec_value) {
  if (spec_id == sycl::detail::ITTSpecConstId)
    HasITTEnabled = true;

  return PI_SUCCESS;
}

static void reset() {
  using namespace sycl::detail;
  HasITTEnabled = false;
  SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::reset();
}

TEST(ITTNotify, UseKernelBundle) {
  set_env(ITTProfileEnvVarName, "1");

  reset();

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<
      sycl::detail::PiApiKind::piextProgramSetSpecializationConstant>(
      redefinedProgramSetSpecializationConstant);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel<>>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(HasITTEnabled, true);
}

TEST(ITTNotify, VarNotSet) {
  unset_env(ITTProfileEnvVarName);

  reset();

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<
      sycl::detail::PiApiKind::piextProgramSetSpecializationConstant>(
      redefinedProgramSetSpecializationConstant);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto ExecBundle = sycl::build(KernelBundle);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(ExecBundle);
    CGH.single_task<TestKernel<>>([] {}); // Actual kernel does not matter
  });

  EXPECT_EQ(HasITTEnabled, false);
}
