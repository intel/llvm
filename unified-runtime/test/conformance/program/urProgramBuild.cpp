// Copyright (C) 2023-2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urProgramBuildTest = uur::urProgramTest;
UUR_DEVICE_TEST_SUITE_WITH_DEFAULT_QUEUE(urProgramBuildTest);

TEST_P(urProgramBuildTest, Success) {
  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
}

TEST_P(urProgramBuildTest, SuccessWithOptions) {
  const char *pOptions = "";
  ASSERT_SUCCESS(urProgramBuild(context, program, pOptions));
}

TEST_P(urProgramBuildTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramBuild(nullptr, program, nullptr));
}

TEST_P(urProgramBuildTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramBuild(context, nullptr, nullptr));
}

TEST_P(urProgramBuildTest, BuildFailure) {
  // The build failure we are testing for happens at SYCL compile time on
  // AMD and Nvidia, so no binary exists to check for a build failure
  ur_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(ur_backend_t), &backend, nullptr));
  if (backend == UR_BACKEND_HIP || backend == UR_BACKEND_CUDA ||
      backend == UR_BACKEND_OFFLOAD) {
    GTEST_SKIP()
        << "Build failure test not supported on AMD/Nvidia/Offload yet";
  }
  // TODO: This seems to fail on opencl/device combination used in the Github
  // runners (`2023.16.12.0.12_195853.xmain-hotfix`). It segfaults, so we just
  // skip the test so other tests can run
  if (backend == UR_BACKEND_OPENCL) {
    GTEST_SKIP() << "Skipping opencl build failure test - segfaults on CI";
  }

  ur_program_handle_t program = nullptr;
  std::shared_ptr<std::vector<char>> il_binary;
  UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
      "build_failure", platform, il_binary));

  ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                   urProgramCreateWithIL(context, il_binary->data(),
                                         il_binary->size(), nullptr, &program));
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE,
                   urProgramBuild(context, program, nullptr));
}
