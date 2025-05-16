// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urProgramCreateWithILTest : uur::urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    // TODO: This should use a query for urProgramCreateWithIL support or
    // rely on UR_RESULT_ERROR_UNSUPPORTED_FEATURE being returned.
    ur_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(ur_backend_t), &backend, nullptr));
    if (backend == UR_BACKEND_HIP) {
      GTEST_SKIP();
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::KernelsEnvironment::instance->LoadSource(
        "foo", platform, il_binary));
  }

  void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
  }

  std::shared_ptr<std::vector<char>> il_binary;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urProgramCreateWithILTest);

TEST_P(urProgramCreateWithILTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::OpenCL{"gfx1100"});

  ur_program_handle_t program = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urProgramCreateWithIL(
      context, il_binary->data(), il_binary->size(), nullptr, &program));
  ASSERT_NE(nullptr, program);
  ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramCreateWithILTest, SuccessWithProperties) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::OpenCL{"gfx1100"});

  std::string string = "test metadata";
  ur_program_metadata_value_t md_value_string;
  md_value_string.pString = string.data();
  ur_program_metadata_t meta_string = {string.data(),
                                       UR_PROGRAM_METADATA_TYPE_STRING,
                                       string.size(), md_value_string};

  ur_program_metadata_value_t md_value_32;
  md_value_32.data32 = 32;
  ur_program_metadata_t meta_32 = {string.data(),
                                   UR_PROGRAM_METADATA_TYPE_UINT32,
                                   sizeof(uint32_t), md_value_32};

  ur_program_metadata_value_t md_value_64;
  md_value_64.data64 = 64;
  ur_program_metadata_t meta_64 = {string.data(),
                                   UR_PROGRAM_METADATA_TYPE_UINT64,
                                   sizeof(uint64_t), md_value_64};

  ur_program_metadata_value_t md_value_data;
  std::vector<uint8_t> metadataValue = {0xDE, 0xAD, 0xBE, 0xEF};
  md_value_data.pData = metadataValue.data();
  ur_program_metadata_t meta_data = {string.data(),
                                     UR_PROGRAM_METADATA_TYPE_BYTE_ARRAY,
                                     metadataValue.size(), md_value_data};

  std::vector<ur_program_metadata_t> metadatas = {meta_string, meta_32, meta_64,
                                                  meta_data};

  ur_program_properties_t properties{
      UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
      static_cast<uint32_t>(metadatas.size()), metadatas.data()};

  ur_program_handle_t program = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urProgramCreateWithIL(
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

TEST_P(urProgramCreateWithILTest, InvalidNullPointer) {

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramCreateWithIL(context, nullptr, il_binary->size(),
                                         nullptr, nullptr));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramCreateWithIL(context, il_binary->data(),
                                         il_binary->size(), nullptr, nullptr));

  ur_program_properties_t properties{UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES,
                                     nullptr, 1, nullptr};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramCreateWithIL(context, il_binary->data(),
                                         il_binary->size(), &properties,
                                         nullptr));
}

TEST_P(urProgramCreateWithILTest, InvalidSize) {
  ur_program_handle_t program = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urProgramCreateWithIL(context, il_binary->data(), 0, nullptr, &program));

  std::string md_string = "test metadata";
  ur_program_metadata_value_t md_value = {};
  md_value.pString = md_string.data();
  ur_program_metadata_t metadata = {md_string.data(),
                                    UR_PROGRAM_METADATA_TYPE_STRING,
                                    md_string.size(), md_value};

  ur_program_properties_t properties{UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES,
                                     nullptr, 0, &metadata};

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urProgramCreateWithIL(context, il_binary->data(), 1,
                                         &properties, &program));
}

TEST_P(urProgramCreateWithILTest, InvalidBinary) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_program_handle_t program = nullptr;
  char binary[] = {0, 1, 2, 3, 4};
  auto result = urProgramCreateWithIL(context, &binary, 5, nullptr, &program);
  // The driver is not required to reject the binary
  ASSERT_TRUE(result == UR_RESULT_ERROR_INVALID_BINARY ||
              result == UR_RESULT_SUCCESS ||
              result == UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE);
}

TEST_P(urProgramCreateWithILTest, CompilerNotAvailable) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_bool_t compiler_available = false;
  urDeviceGetInfo(device, UR_DEVICE_INFO_COMPILER_AVAILABLE, sizeof(ur_bool_t),
                  &compiler_available, nullptr);

  if (!compiler_available) {
    ur_program_handle_t program = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE,
                     urProgramCreateWithIL(context, il_binary->data(), 1,
                                           nullptr, &program));
  }
}
