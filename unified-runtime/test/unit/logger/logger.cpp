// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fstream>
#include <sstream>
#include <thread>

#include "fixtures.hpp"
#include "logger/ur_logger_details.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST_F(DefaultLoggerWithFileSink, DefaultLevelNoOutput) {
  URLOG_(*logger, INFO, "This should not be printed: {}", 42);
  test_msg.clear();
}

TEST_F(DefaultLoggerWithFileSink, MultipleLines) {
  URLOG_(*logger, WARN, "Test message: {}", "success");
  URLOG_(*logger, DEBUG, "This should not be printed: {}", 42);
  URLOG_(*logger, ERR, "Test message: {}", "success");

  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n"
           << test_msg_prefix << "[ERROR]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, ThreeParams) {
  URLOG_(*logger, ERR, "{} {}: {}", "Test", 42, 3.8);
  test_msg << test_msg_prefix << "[ERROR]: Test 42: 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces) {
  URLOG_(*logger, ERR, "{{}} {}: {}", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: {} Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces2) {
  URLOG_(*logger, ERR, "200 {{ {}: {{{}}} 3.8", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: 200 { Test: {42} 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces3) {
  URLOG_(*logger, ERR, "{{ {}:}} {}}}", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: { Test:} 42}\n";
}

TEST_F(DefaultLoggerWithFileSink, NoBraces) {
  URLOG_(*logger, ERR, " Test: 42");
  test_msg << test_msg_prefix << "[ERROR]:  Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelDebug) {
  auto level = logger::Level::DEBUG;
  logger->setLevel(level);
  logger->setFlushLevel(level);
  URLOG_(*logger, DEBUG, "Test message: {}", "success");

  test_msg << test_msg_prefix << "[DEBUG]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelInfo) {
  auto level = logger::Level::INFO;
  logger->setLevel(level);
  logger->setFlushLevel(level);
  URLOG_(*logger, INFO, "Test message: {}", "success");
  URLOG_(*logger, DEBUG, "This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[INFO]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelWarning) {
  auto level = logger::Level::WARN;
  logger->setLevel(level);
  URLOG_(*logger, WARN, "Test message: {}", "success");
  URLOG_(*logger, INFO, "This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelError) {
  logger->setLevel(logger::Level::ERR);
  URLOG_(*logger, ERR, "Test message: {}", "success");
  URLOG_(*logger, WARN, "This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[ERROR]: Test message: success\n";
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(UniquePtrLoggerWithFilesink, SetLogLevelAndFlushLevelDebugWithCtor) {
  auto level = logger::Level::DEBUG;
  logger = std::make_unique<logger::Logger>(
      level, std::make_unique<logger::FileSink>(logger_name, file_path, level));

  URLOG_(*logger, DEBUG, "Test message: {}", "success");
  test_msg << test_msg_prefix << "[DEBUG]: Test message: success\n";
}

TEST_F(UniquePtrLoggerWithFilesink, NestedFilePath) {
  auto dir_name = "tmp_dir";
  file_path = dir_name;
  for (int i = 0; i < 20; ++i) {
    file_path /= dir_name;
  }
  filesystem::create_directories(file_path);
  file_path /= file_name;
  logger = std::make_unique<logger::Logger>(
      logger::Level::WARN, std::make_unique<logger::FileSink>(
                               logger_name, file_path, logger::Level::WARN));

  URLOG_(*logger, WARN, "Test message: {}", "success");
  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n";
}

TEST_F(UniquePtrLoggerWithFilesinkFail, NullSink) {
  logger = std::make_unique<logger::Logger>(logger::Level::INFO, nullptr);
  URLOG_(*logger, INFO, "This should not be printed: {}", 42);
  test_msg.clear();
}

//////////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ThreadCount, FileSinkLoggerMultipleThreads,
    ::testing::Values(2 * std::thread::hardware_concurrency(),
                      std::thread::hardware_concurrency(),
                      std::thread::hardware_concurrency() / 2));
TEST_P(FileSinkLoggerMultipleThreads, Multithreaded) {
  std::vector<std::thread> threads;
  auto local_logger = logger::Logger(
      logger::Level::WARN,
      std::make_unique<logger::FileSink>(logger_name, file_path, true));
  constexpr int message_count = 50;

  // Messages below the flush level
  for (int i = 0; i < thread_count; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < message_count; ++j) {
        URLOG_(local_logger, WARN, "Test message: {}", "it's a success");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
  threads.clear();

  // Messages at the flush level
  for (int i = 0; i < thread_count; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < message_count; ++j) {
        URLOG_(local_logger, ERR, "Flushed test message: {}", "it's a success");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (int i = 0; i < thread_count * message_count; ++i) {
    test_msg << "Test message: it's a success\n";
  }
  for (int i = 0; i < thread_count * message_count; ++i) {
    test_msg << "Flushed test message: it's a success\n";
  }
}

//////////////////////////////////////////////////////////////////////////////
INSTANTIATE_TEST_SUITE_P(
    ThreadCount, CommonLoggerWithMultipleThreads,
    ::testing::Values(2, std::thread::hardware_concurrency()));
TEST_P(CommonLoggerWithMultipleThreads, StdoutMultithreaded) {
  std::vector<std::thread> threads;
  auto local_logger = logger::Logger(
      logger::Level::WARN, std::make_unique<logger::StdoutSink>("test", true));
  constexpr int message_count = 50;

  // Messages below the flush level
  for (int i = 0; i < thread_count; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < message_count; ++j) {
        URLOG_(local_logger, WARN, "Test message: {}", "it's a success");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
  threads.clear();

  // Messages at the flush level
  for (int i = 0; i < thread_count; i++) {
    threads.emplace_back([&]() {
      for (int j = 0; j < message_count; ++j) {
        URLOG_(local_logger, ERR, "Flushed test message: {}", "it's a success");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
