// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <uur/fixtures.h>

using urLevelZeroProgramLinkTest = uur::urProgramTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urLevelZeroProgramLinkTest);

TEST_P(urLevelZeroProgramLinkTest, InvalidLinkOptionsPrintedInLog) {
  ur_program_handle_t linked_program = nullptr;
  ASSERT_SUCCESS(urProgramCompile(context, program, "-foo"));
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_PROGRAM_LINK_FAILURE,
      urProgramLink(context, 1, &program, "-foo", &linked_program));

  size_t logSize;
  std::vector<char> log;

  ASSERT_SUCCESS(urProgramGetBuildInfo(
      linked_program, device, UR_PROGRAM_BUILD_INFO_LOG, 0, nullptr, &logSize));
  log.resize(logSize);
  log[logSize - 1] = 'x';
  ASSERT_SUCCESS(urProgramGetBuildInfo(linked_program, device,
                                       UR_PROGRAM_BUILD_INFO_LOG, logSize,
                                       log.data(), nullptr));
  ASSERT_EQ(log[logSize - 1], '\0');
  ASSERT_NE(std::string{log.data()}.find("-foo"), std::string::npos);

  ASSERT_SUCCESS(urProgramRelease(linked_program));
}
