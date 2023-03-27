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
TEST_F(LoggerWithFileSink, DefaultLevelNoOutput) {
    logger->info("This should not be printed: {}", 42);
    test_msg.clear();
}

TEST_F(LoggerWithFileSink, MultipleLines) {
    logger->warning("Test message: {}", "success");
    logger->debug("This should not be printed: {}", 42);
    logger->error("Test message: {}", "success");

    test_msg += "[WARNING]: Test message: success\n"
                "<test>[ERROR]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, ThreeParams) {
    logger->error("{} {}: {}", "Test", 42, 3.8);
    test_msg += "[ERROR]: Test 42: 3.8\n";
}

TEST_F(LoggerWithFileSink, DoubleBraces) {
    logger->error("{{}} {}: {}", "Test", 42);
    test_msg += "[ERROR]: {} Test: 42\n";
}

TEST_F(LoggerWithFileSink, DoubleBraces2) {
    logger->error("200 {{ {}: {{{}}} 3.8", "Test", 42);
    test_msg += "[ERROR]: 200 { Test: {42} 3.8\n";
}

TEST_F(LoggerWithFileSink, DoubleBraces3) {
    logger->error("{{ {}:}} {}}}", "Test", 42);
    test_msg += "[ERROR]: { Test:} 42}\n";
}

TEST_F(LoggerWithFileSink, NoBraces) {
    logger->error(" Test: 42");
    test_msg += "[ERROR]:  Test: 42\n";
}

TEST_F(LoggerWithFileSink, SetLogLevelAndFlushLevelDebugWithCtor) {
    auto level = logger::Level::DEBUG;
    logger = std::make_unique<logger::Logger>(
        level,
        std::make_unique<logger::FileSink>(logger_name, file_path, level));

    logger->debug("Test message: {}", "success");
    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, SetLevelDebug) {
    auto level = logger::Level::DEBUG;
    logger->setLevel(level);
    logger->setFlushLevel(level);
    logger->debug("Test message: {}", "success");

    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, SetLevelInfo) {
    auto level = logger::Level::INFO;
    logger->setLevel(level);
    logger->setFlushLevel(level);
    logger->info("Test message: {}", "success");
    logger->debug("This should not be printed: {}", 42);

    test_msg += "[INFO]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, SetLevelWarning) {
    auto level = logger::Level::WARN;
    logger->setLevel(level);
    logger->warning("Test message: {}", "success");
    logger->info("This should not be printed: {}", 42);

    test_msg += "[WARNING]: Test message: success\n";
}

TEST_F(LoggerWithFileSink, SetLevelError) {
    logger->setLevel(logger::Level::ERR);
    logger->error("Test message: {}", "success");
    logger->warning("This should not be printed: {}", 42);

    test_msg += "[ERROR]: Test message: success\n";
}
