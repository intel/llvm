// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#ifndef UR_UNIT_LOGGER_TEST_FIXTURES_HPP
#define UR_UNIT_LOGGER_TEST_FIXTURES_HPP

#include <gtest/gtest.h>

#include "logger/ur_logger.hpp"

class LoggerFromEnvVar : public ::testing::Test {
protected:
  std::string logger_name = "ADAPTER_TEST";

  void SetUp() override { logger::init(logger_name); }
};

class LoggerCommonSetup : public ::testing::Test {
protected:
  const std::string logger_name = "test";
  const std::string test_msg_prefix = "<" + logger_name + ">";
};

class LoggerWithFileSink : public LoggerCommonSetup {
protected:
  const filesystem::path file_name = "ur_test_logger.log";
  filesystem::path file_path = file_name;
  std::stringstream test_msg;

  void TearDown() override {
    auto test_log = std::ifstream(file_path);
    ASSERT_TRUE(test_log.good());
    std::stringstream printed_msg;
    printed_msg << test_log.rdbuf();
    test_log.close();

    ASSERT_GT(filesystem::remove_all(*file_path.begin()), 0);
    ASSERT_EQ(printed_msg.str(), test_msg.str());
  }
};

class LoggerWithFileSinkFail : public LoggerWithFileSink {
protected:
  void TearDown() override {
    auto test_log = std::ifstream(file_path);
    ASSERT_FALSE(test_log.good());
  }
};

class CommonLoggerWithMultipleThreads
    : public LoggerCommonSetup,
      public ::testing::WithParamInterface<int> {
protected:
  int thread_count;

  void SetUp() override { thread_count = GetParam(); }
};

class FileSinkLoggerMultipleThreads
    : public LoggerWithFileSink,
      public ::testing::WithParamInterface<int> {
protected:
  int thread_count;

  void SetUp() override { thread_count = GetParam(); }
};

class UniquePtrLoggerWithFilesink : public LoggerWithFileSink {
protected:
  std::unique_ptr<logger::Logger> logger;

  void TearDown() override {
    logger.reset();
    LoggerWithFileSink::TearDown();
  }
};

class UniquePtrLoggerWithFilesinkFail : public LoggerWithFileSinkFail {
protected:
  std::unique_ptr<logger::Logger> logger;

  void TearDown() override {
    logger.reset();
    LoggerWithFileSinkFail::TearDown();
  }
};

class DefaultLoggerWithFileSink : public UniquePtrLoggerWithFilesink {
protected:
  void SetUp() override {
    logger = std::make_unique<logger::Logger>(
        logger::Level::WARN,
        std::make_unique<logger::FileSink>(logger_name, file_path));
  }
};

#endif // UR_UNIT_LOGGER_TEST_FIXTURES_HPP
