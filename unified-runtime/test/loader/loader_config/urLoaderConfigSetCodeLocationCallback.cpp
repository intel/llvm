// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

ur_code_location_t codeLocationCallback([[maybe_unused]] void *userData) {
  ur_code_location_t codeloc;
  codeloc.columnNumber = 1;
  codeloc.lineNumber = 2;
  codeloc.functionName = "fname";
  codeloc.sourceFile = "sfile";

  return codeloc;
}

struct urLoaderConfigSetCodeLocationCallbackTest : LoaderConfigTest {};

TEST_F(urLoaderConfigSetCodeLocationCallbackTest, Success) {
  ASSERT_SUCCESS(urLoaderConfigSetCodeLocationCallback(
      loaderConfig, codeLocationCallback, nullptr));
}

TEST_F(urLoaderConfigSetCodeLocationCallbackTest, InvalidNullHandle) {
  ASSERT_EQ(urLoaderConfigSetCodeLocationCallback(nullptr, codeLocationCallback,
                                                  nullptr),
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_F(urLoaderConfigSetCodeLocationCallbackTest, InvalidNullPointer) {
  ASSERT_EQ(
      urLoaderConfigSetCodeLocationCallback(loaderConfig, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
