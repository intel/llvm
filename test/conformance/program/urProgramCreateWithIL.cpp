// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urProgramCreateWithILTest : uur::urContextTest {
  void SetUp() override {
    // We haven't got device code tests working on native cpu yet.
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    // TODO: This should use a query for urProgramCreateWithIL support or
    // rely on UR_RESULT_ERROR_UNSUPPORTED_FEATURE being returned.
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(ur_platform_backend_t), &backend,
                                     nullptr));
    if (backend == UR_PLATFORM_BACKEND_HIP) {
      GTEST_SKIP();
    }
    uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);
  }

  void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
  }

  std::shared_ptr<std::vector<char>> il_binary;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramCreateWithILTest);

TEST_P(urProgramCreateWithILTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_program_handle_t program = nullptr;
  ASSERT_SUCCESS(urProgramCreateWithIL(context, il_binary->data(),
                                       il_binary->size(), nullptr, &program));
  ASSERT_NE(nullptr, program);
  ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramCreateWithILTest, SuccessWithProperties) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_program_properties_t properties{UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES,
                                     nullptr, 0, nullptr};
  ur_program_handle_t program = nullptr;
  ASSERT_SUCCESS(urProgramCreateWithIL(
      context, il_binary->data(), il_binary->size(), &properties, &program));
  ASSERT_NE(nullptr, program);
  ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullHandle) {
  ur_program_handle_t program = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramCreateWithIL(nullptr, il_binary->data(),
                                         il_binary->size(), nullptr, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullPointerSource) {
  ur_program_handle_t program = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramCreateWithIL(context, nullptr, il_binary->size(),
                                         nullptr, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidSizeLength) {
  ur_program_handle_t program = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urProgramCreateWithIL(context, il_binary->data(), 0, nullptr, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidNullPointerProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramCreateWithIL(context, il_binary->data(),
                                         il_binary->size(), nullptr, nullptr));
}

TEST_P(urProgramCreateWithILTest, BuildInvalidProgram) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_program_handle_t program = nullptr;
  char binary[] = {0, 1, 2, 3, 4};
  auto result = urProgramCreateWithIL(context, &binary, 5, nullptr, &program);
  // The driver is not required to reject the binary
  ASSERT_TRUE(result == UR_RESULT_ERROR_INVALID_BINARY ||
              result == UR_RESULT_SUCCESS);
}
