// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "helpers.h"
#include "logger/ur_logger.hpp"
#include "ur_util.hpp"

TEST(logger, NullSinkOneParam) {
    ASSERT_THROW(logger::Logger(nullptr), std::invalid_argument);
}

TEST(logger, NullSinkTwoParams) {
    ASSERT_THROW(logger::Logger(logger::Level::ERR, nullptr),
                 std::invalid_argument);
}

class FileSink : public ::testing::Test {
  protected:
    std::string file_path = "ur_test_logger.log";
    std::string test_msg = "";

    void TearDown() override {
        auto test_log = std::ifstream(file_path, std::ios::in);
        ASSERT_TRUE(test_log.good());
        std::stringstream printed_msg;
        printed_msg << test_log.rdbuf();
        test_log.close();

        std::remove(file_path.c_str());
        ASSERT_EQ(printed_msg.str(), test_msg);
    }
};

class FileSinkDefaultLevel : public FileSink {
  protected:
    void SetUp() override {
        logger::logger = std::make_unique<logger::Logger>(
            std::make_unique<logger::FileSink>(file_path));
    }
};

TEST_F(FileSink, MultipleLines) {
    logger::Level level = logger::Level::WARN;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(file_path));

    logger.warning("Test message: {}", "success");
    logger.debug("This should not be printed: {}", 42);
    logger.error("Test message: {}", "success");

    test_msg = "[WARNING]:Test message: success\n"
               "[ERROR]:Test message: success\n";
}

TEST_F(FileSink, ThreeParams) {
    logger::Level level = logger::Level::DEBUG;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(file_path));

    logger.setFlushLevel(level);
    logger.debug("{} {}: {}", "Test", 42, 3.8);
    test_msg = "[DEBUG]:Test 42: 3.8\n";
}

TEST_F(FileSink, DoubleBraces) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(file_path));

    logger.error("{{}} {}: {}", "Test", 42);
    test_msg = "[ERROR]:{} Test: 42\n";
}

TEST_F(FileSink, DoubleBraces2) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(file_path));

    logger.error("200 {{ {}: {{{}}} 3.8", "Test", 42);
    test_msg = "[ERROR]:200 { Test: {42} 3.8\n";
}

TEST_F(FileSink, DoubleBraces3) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(file_path));

    logger.error("{{ {}:}} {}}}", "Test", 42);
    test_msg = "[ERROR]:{ Test:} 42}\n";
}

TEST_F(FileSink, NoBraces) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(file_path));

    logger.error(" Test: 42");
    test_msg = "[ERROR]: Test: 42\n";
}

TEST_F(FileSink, SetFlushLevelDebugCtor) {
    auto level = logger::Level::DEBUG;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(file_path, level));

    logger.debug("Test message: {}", "success");
    test_msg = "[DEBUG]:Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, DefaultLevelNoOutput) {
    logger::logger->debug("This should not be printed: {}", 42);
}

TEST_F(FileSinkDefaultLevel, SetLevelDebug) {
    auto level = logger::Level::DEBUG;
    logger::logger->setLevel(level);
    logger::logger->setFlushLevel(level);
    logger::logger->debug("Test message: {}", "success");

    test_msg = "[DEBUG]:Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelInfo) {
    auto level = logger::Level::INFO;
    logger::logger->setLevel(level);
    logger::logger->setFlushLevel(level);
    logger::logger->info("Test message: {}", "success");
    logger::logger->debug("This should not be printed: {}", 42);

    test_msg = "[INFO]:Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelWarning) {
    auto level = logger::Level::WARN;
    logger::logger->setLevel(level);
    logger::logger->setFlushLevel(level);
    logger::logger->warning("Test message: {}", "success");
    logger::logger->info("This should not be printed: {}", 42);

    test_msg = "[WARNING]:Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelError) {
    logger::logger->setLevel(logger::Level::ERR);
    logger::logger->error("Test message: {}", "success");
    logger::logger->warning("This should not be printed: {}", 42);

    test_msg = "[ERROR]:Test message: success\n";
}
