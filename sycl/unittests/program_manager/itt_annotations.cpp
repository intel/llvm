//==---- itt_annotations.cpp --- ITT Annotations for SPIR-V images ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/config.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>

#include <helpers/TestKernel.hpp>

using namespace sycl::unittest;

bool HasITTEnabled = false;

static ur_result_t redefinedProgramSetSpecializationConstants(void *pParams) {
  auto params =
      *static_cast<ur_program_set_specialization_constants_params_t *>(pParams);
  for (uint32_t SpecConstIndex = 0; SpecConstIndex < *params.pcount;
       SpecConstIndex++) {
    if ((*params.ppSpecConstants)[SpecConstIndex].id ==
        sycl::detail::ITTSpecConstId)
      HasITTEnabled = true;
  }

  return UR_RESULT_SUCCESS;
}

TEST(ITTNotify, UseKernelBundle) {
  ScopedEnvVar Var("INTEL_ENABLE_OFFLOAD_ANNOTATIONS", "1",
                   SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::reset);
  HasITTEnabled = false;

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback(
      "urProgramSetSpecializationConstants",
      &redefinedProgramSetSpecializationConstants);

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
  ScopedEnvVar Var("INTEL_ENABLE_OFFLOAD_ANNOTATIONS", nullptr,
                   SYCLConfig<INTEL_ENABLE_OFFLOAD_ANNOTATIONS>::reset);
  HasITTEnabled = false;
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback(
      "urProgramSetSpecializationConstants",
      &redefinedProgramSetSpecializationConstants);

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
