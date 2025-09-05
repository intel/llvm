// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urProgramCompileWithParamTest = uur::urProgramTestWithParam<std::string>;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(urProgramCompileWithParamTest,
                                 ::testing::Values("-O0", "-O1", "-O2", "-O3"),
                                 uur::deviceTestWithParamPrinter<std::string>);

TEST_P(urProgramCompileWithParamTest, Success) {
  const char *platformOption = nullptr;
  ASSERT_SUCCESS(urPlatformGetBackendOption(platform, getParam().c_str(),
                                            &platformOption));

  ASSERT_SUCCESS(urProgramCompile(context, program, platformOption));
}

using urProgramCompileTest = uur::urProgramTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urProgramCompileTest);

TEST_P(urProgramCompileTest, Success) {
  ASSERT_SUCCESS(urProgramCompile(context, program, nullptr));
}

TEST_P(urProgramCompileTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramCompile(nullptr, program, nullptr));
}

TEST_P(urProgramCompileTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramCompile(context, nullptr, nullptr));
}
