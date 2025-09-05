/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_options.cpp
 *
 */

// RUN: UR_LOG_SANITIZER="level:debug;flush:debug;output:stdout" sanitizer_options-test
// REQUIRES: sanitizer

#include "sanitizer_options.hpp"
#include "sanitizer_options_impl.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>

using namespace ur_sanitizer_layer;
using namespace ur_sanitizer_layer::options;

struct OptionParserTest : public ::testing::Test {
  logger::Logger Logger;
  EnvVarMap EnvMap;
  OptionParser Parser;

  OptionParserTest()
      : Logger(logger::create_logger("OptionParser", false, false,
                                     UR_LOGGER_LEVEL_DEBUG)),
        Parser(EnvMap, Logger) {}
};

TEST_F(OptionParserTest, ParseBool_Normal) {
  bool Result;

  EnvMap["test"] = {"1"};
  Result = false;
  Parser.ParseBool("test", Result);
  ASSERT_TRUE(Result);

  EnvMap["test"] = {"TrUe"};
  Result = false;
  Parser.ParseBool("test", Result);
  ASSERT_TRUE(Result);

  EnvMap["test"] = {"0"};
  Result = true;
  Parser.ParseBool("test", Result);
  ASSERT_FALSE(Result);

  EnvMap["test"] = {"False"};
  Result = true;
  Parser.ParseBool("test", Result);
  ASSERT_FALSE(Result);
}

TEST_F(OptionParserTest, ParseBool_Default) {
  bool Result;

  Result = false;
  Parser.ParseBool("null", Result);
  ASSERT_FALSE(Result);

  Result = true;
  Parser.ParseBool("null", Result);
  ASSERT_TRUE(Result);
}

TEST_F(OptionParserTest, ParseBool_Error) {
  bool Result;

  EnvMap["test"] = {"42"};
  ASSERT_DEATH(Parser.ParseBool("test", Result), ".*");
}

TEST_F(OptionParserTest, ParseUint64_Normal) {
  uint64_t Result;

  EnvMap["test"] = {"42"};
  Parser.ParseUint64("test", Result);
  ASSERT_EQ(Result, 42);
}

TEST_F(OptionParserTest, ParseUint64_Default) {
  uint64_t Result;

  Result = 42;
  Parser.ParseUint64("null", Result);
  ASSERT_EQ(Result, 42);
}

TEST_F(OptionParserTest, ParseUint64_Error) {
  uint64_t Result;

  EnvMap["test"] = {"-42"};
  ASSERT_DEATH(Parser.ParseUint64("test", Result), ".*");

  EnvMap["test"] = {"abc"};
  ASSERT_DEATH(Parser.ParseUint64("test", Result), ".*");
}

TEST_F(OptionParserTest, ParseUint64_OutOfRange) {
  uint64_t Result;

  EnvMap["test"] = {"100"};
  Parser.ParseUint64("test", Result, 0, 99);
  ASSERT_EQ(Result, 99);

  EnvMap["test"] = {"1"};
  Parser.ParseUint64("test", Result, 10, 11);
  ASSERT_EQ(Result, 10);
}

struct SanitizerOptionsTest : public ::testing::Test {
  logger::Logger Logger;
  std::string EnvName = "SANITIZER_OPTIONS_TEST";
  SanitizerOptions Options;

  SanitizerOptionsTest()
      : Logger(logger::create_logger("SanitizerOptions", false, false,
                                     UR_LOGGER_LEVEL_DEBUG)) {}

  void SetEnvAndInit(const std::string &Value) {
    setenv(EnvName.c_str(), Value.c_str(), 1);
    Options.Init(EnvName, Logger);
  }
};

TEST_F(SanitizerOptionsTest, Default) {
  SetEnvAndInit("");
  ASSERT_FALSE(Options.Debug);
}

TEST_F(SanitizerOptionsTest, Normal) {
  SetEnvAndInit("debug:true");
  ASSERT_TRUE(Options.Debug);

  SetEnvAndInit("debug:false");
  ASSERT_FALSE(Options.Debug);
}

TEST_F(SanitizerOptionsTest, Error) {
  // invalid format for bool
  ASSERT_DEATH(SetEnvAndInit("debug:42"), "for enable");
  ASSERT_DEATH(SetEnvAndInit("debug:yes"), "for enable");

  // invalid format for uint64
  ASSERT_DEATH(SetEnvAndInit("quarantine_size_mb:-1"), "The valid range of");
  ASSERT_DEATH(SetEnvAndInit("quarantine_size_mb:abc"), "The valid range of");

  // out of range error will not result in death, but will clamp to the valid
  // range. For MaxQuarantineSizeMB, its valid range is [0, UINT32_MAX]
  SetEnvAndInit("quarantine_size_mb:4294967296");
  ASSERT_EQ(Options.MaxQuarantineSizeMB, UINT32_MAX);

  // invalid format for UR env map
  ASSERT_DEATH(SetEnvAndInit("debug=42"), "Proper format is");
  ASSERT_DEATH(SetEnvAndInit("a:1,b:1"), "Proper format is");
}
