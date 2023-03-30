// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>

#include "fixtures.hpp"
#include "logger/ur_logger_details.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST(LoggerFailure, NullSinkOneParam) {
    ASSERT_THROW(logger::Logger(nullptr), std::invalid_argument);
}

TEST(LoggerFailure, NullSinkTwoParams) {
    ASSERT_THROW(logger::Logger(logger::Level::ERR, nullptr),
                 std::invalid_argument);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(LoggerFromEnvVar, BasicMessage) {
    logger::info("Test message: {}", "success");
    logger::debug("This should not be printed: {}", 42);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DefaultLoggerWithFileSink, DefaultLevelNoOutput) {
    logger->info("This should not be printed: {}", 42);
    test_msg.clear();
}

TEST_F(DefaultLoggerWithFileSink, MultipleLines) {
    logger->warning("Test message: {}", "success");
    logger->debug("This should not be printed: {}", 42);
    logger->error("Test message: {}", "success");

    test_msg += "[WARNING]: Test message: success\n"
                "<test>[ERROR]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, ThreeParams) {
    logger->error("{} {}: {}", "Test", 42, 3.8);
    test_msg += "[ERROR]: Test 42: 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces) {
    logger->error("{{}} {}: {}", "Test", 42);
    test_msg += "[ERROR]: {} Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces2) {
    logger->error("200 {{ {}: {{{}}} 3.8", "Test", 42);
    test_msg += "[ERROR]: 200 { Test: {42} 3.8\n";
}

TEST_F(DefaultLoggerWithFileSink, DoubleBraces3) {
    logger->error("{{ {}:}} {}}}", "Test", 42);
    test_msg += "[ERROR]: { Test:} 42}\n";
}

TEST_F(DefaultLoggerWithFileSink, NoBraces) {
    logger->error(" Test: 42");
    test_msg += "[ERROR]:  Test: 42\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelDebug) {
    auto level = logger::Level::DEBUG;
    logger->setLevel(level);
    logger->setFlushLevel(level);
    logger->debug("Test message: {}", "success");

    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelInfo) {
    auto level = logger::Level::INFO;
    logger->setLevel(level);
    logger->setFlushLevel(level);
    logger->info("Test message: {}", "success");
    logger->debug("This should not be printed: {}", 42);

    test_msg += "[INFO]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelWarning) {
    auto level = logger::Level::WARN;
    logger->setLevel(level);
    logger->warning("Test message: {}", "success");
    logger->info("This should not be printed: {}", 42);

    test_msg += "[WARNING]: Test message: success\n";
}

TEST_F(DefaultLoggerWithFileSink, SetLevelError) {
    logger->setLevel(logger::Level::ERR);
    logger->error("Test message: {}", "success");
    logger->warning("This should not be printed: {}", 42);

    test_msg += "[ERROR]: Test message: success\n";
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(LoggerWithFileSink, SetLogLevelAndFlushLevelDebugWithCtor) {
    auto level = logger::Level::DEBUG;
    logger = std::make_unique<logger::Logger>(
        level,
        std::make_unique<logger::FileSink>(logger_name, file_path, level));

    logger->debug("Test message: {}", "success");
    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, NestedFilePath) {
    auto dir_name = "tmp_dir";
    file_path = dir_name;
    for (int i = 0; i < 20; ++i) {
        file_path /= dir_name;
    }
    std::filesystem::create_directories(file_path);
    file_path /= file_name;
    logger = std::make_unique<logger::Logger>(
        logger::Level::WARN,
        std::make_unique<logger::FileSink>(logger_name, file_path));

    logger->warning("Test message: {}", "success");
    test_msg += "[WARNING]: Test message: success\n";
}
