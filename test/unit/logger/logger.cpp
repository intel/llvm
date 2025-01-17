// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <fstream>
#include <sstream>
#include <thread>

#include "fixtures.hpp"
#include "logger/ur_logger_details.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST_F(DefaultLoggerWithFileSink, DefaultLevelNoOutput) {
  logger->info("This should not be printed: {}", 42);
  test_msg.clear();
}

TEST_F(DefaultLoggerWithFileSink, MultipleLines) {
  logger->warning("Test message: {}", "success");
  logger->debug("This should not be printed: {}", 42);
  logger->error("Test message: {}", "success");

  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n"
           << test_msg_prefix << "[ERROR]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, ThreeParams) {
  logger->error("{} {}: {}", "Test", 42, 3.8);
  test_msg << test_msg_prefix << "[ERROR]: Test 42: 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces) {
  logger->error("{{}} {}: {}", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: {} Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces2) {
  logger->error("200 {{ {}: {{{}}} 3.8", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: 200 { Test: {42} 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces3) {
  logger->error("{{ {}:}} {}}}", "Test", 42);
  test_msg << test_msg_prefix << "[ERROR]: { Test:} 42}\n";
}

TEST_F(DefaultLoggerWithFileSink, NoBraces) {
  logger->error(" Test: 42");
  test_msg << test_msg_prefix << "[ERROR]:  Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelDebug) {
  auto level = logger::Level::DEBUG;
  logger->setLevel(level);
  logger->setFlushLevel(level);
  logger->debug("Test message: {}", "success");

  test_msg << test_msg_prefix << "[DEBUG]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelInfo) {
  auto level = logger::Level::INFO;
  logger->setLevel(level);
  logger->setFlushLevel(level);
  logger->info("Test message: {}", "success");
  logger->debug("This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[INFO]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelWarning) {
  auto level = logger::Level::WARN;
  logger->setLevel(level);
  logger->warning("Test message: {}", "success");
  logger->info("This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelError) {
  logger->setLevel(logger::Level::ERR);
  logger->error("Test message: {}", "success");
  logger->warning("This should not be printed: {}", 42);

  test_msg << test_msg_prefix << "[ERROR]: Test message: success\n";
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(UniquePtrLoggerWithFilesink, SetLogLevelAndFlushLevelDebugWithCtor) {
  auto level = logger::Level::DEBUG;
  logger = std::make_unique<logger::Logger>(
      level, std::make_unique<logger::FileSink>(logger_name, file_path, level));

  logger->debug("Test message: {}", "success");
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

  logger->warning("Test message: {}", "success");
  test_msg << test_msg_prefix << "[WARNING]: Test message: success\n";
}

TEST_F(UniquePtrLoggerWithFilesinkFail, NullSink) {
  logger = std::make_unique<logger::Logger>(logger::Level::INFO, nullptr);
  logger->info("This should not be printed: {}", 42);
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
        local_logger.warn("Test message: {}", "it's a success");
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
        local_logger.error("Flushed test message: {}", "it's a success");
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
        local_logger.warn("Test message: {}", "it's a success");
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
        local_logger.error("Flushed test message: {}", "it's a success");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
