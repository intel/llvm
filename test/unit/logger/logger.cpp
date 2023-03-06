// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "helpers.h"
#include "logger/ur_logger.hpp"
#include "logger/ur_logger_details.hpp"

TEST(logger, NullSinkOneParam) {
    ASSERT_THROW(logger::Logger(nullptr), std::invalid_argument);
}

TEST(logger, NullSinkTwoParams) {
    ASSERT_THROW(logger::Logger(logger::Level::ERR, nullptr),
                 std::invalid_argument);
}

class CreateLoggerWithEnvVar : public ::testing::TestWithParam<std::string> {
  protected:
    int ret = -1;
    std::string env_var_value;

    void SetUp() override {
        env_var_value = GetParam();
        ret = setenv("UR_LOG_TEST_ADAPTER", env_var_value.c_str(), 1);
        ASSERT_EQ(ret, 0);
        logger::init("test_adapter");
        logger::info("{} initialized successfully!", "test_adapter");
    }

    void TearDown() override {
        ret = unsetenv("UR_LOG_TEST_ADAPTER");
        ASSERT_EQ(ret, 0);
    }
};

INSTANTIATE_TEST_SUITE_P(EnvVarSetupStdParams, CreateLoggerWithEnvVar,
                         ::testing::Values("level:info",
                                           "level:info;output:stderr"));

TEST_P(CreateLoggerWithEnvVar, EnvVarSetupStd) {
    logger::info("Test message: {}", "success");
    logger::debug("This should not be printed: {}", 42);
}

class FileSink : public ::testing::Test {
  protected:
    std::string file_path = "ur_test_logger.log";
    std::string logger_name = "test";
    std::string test_msg = "<" + logger_name + ">";

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
    int ret = -1;
    std::string env_var_value = "output:file," + file_path;

    void SetUp() override {
        ret = setenv("UR_LOG_TEST_ADAPTER", env_var_value.c_str(), 1);
        ASSERT_EQ(ret, 0);
        logger::init("test_adapter");
        logger::info("{} initialized successfully!", "test_adapter");
    }

    void TearDown() override {
        ret = unsetenv("UR_LOG_TEST_ADAPTER");
        ASSERT_EQ(ret, 0);
    }
};

TEST_F(FileSink, MultipleLines) {
    logger::Level level = logger::Level::WARN;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.warning("Test message: {}", "success");
    logger.debug("This should not be printed: {}", 42);
    logger.error("Test message: {}", "success");

    test_msg += "[WARNING]: Test message: success\n"
                "<test>[ERROR]: Test message: success\n";
}

TEST_F(FileSink, ThreeParams) {
    logger::Level level = logger::Level::DEBUG;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.setFlushLevel(level);
    logger.debug("{} {}: {}", "Test", 42, 3.8);
    test_msg += "[DEBUG]: Test 42: 3.8\n";
}

TEST_F(FileSink, DoubleBraces) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.error("{{}} {}: {}", "Test", 42);
    test_msg += "[ERROR]: {} Test: 42\n";
}

TEST_F(FileSink, DoubleBraces2) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.error("200 {{ {}: {{{}}} 3.8", "Test", 42);
    test_msg += "[ERROR]: 200 { Test: {42} 3.8\n";
}

TEST_F(FileSink, DoubleBraces3) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.error("{{ {}:}} {}}}", "Test", 42);
    test_msg += "[ERROR]: { Test:} 42}\n";
}

TEST_F(FileSink, NoBraces) {
    logger::Logger logger(logger::Level::ERR,
                          std::make_unique<logger::FileSink>(logger_name, file_path));

    logger.error(" Test: 42");
    test_msg += "[ERROR]:  Test: 42\n";
}

TEST_F(FileSink, SetFlushLevelDebugCtor) {
    auto level = logger::Level::DEBUG;
    logger::Logger logger(level, std::make_unique<logger::FileSink>(logger_name, file_path, level));

    logger.debug("Test message: {}", "success");
    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, DefaultLevelNoOutput) {
    logger::debug("This should not be printed: {}", 42);
}

TEST_F(FileSinkDefaultLevel, SetLevelDebug) {
    auto level = logger::Level::DEBUG;
    logger::setLevel(level);
    logger::setFlushLevel(level);
    logger::debug("Test message: {}", "success");

    test_msg += "[DEBUG]: Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelInfo) {
    auto level = logger::Level::INFO;
    logger::setLevel(level);
    logger::setFlushLevel(level);
    logger::info("Test message: {}", "success");
    logger::debug("This should not be printed: {}", 42);

    test_msg += "[INFO]: Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelWarning) {
    auto level = logger::Level::WARN;
    logger::setLevel(level);
    logger::setFlushLevel(level);
    logger::warning("Test message: {}", "success");
    logger::info("This should not be printed: {}", 42);

    test_msg += "[WARNING]: Test message: success\n";
}

TEST_F(FileSinkDefaultLevel, SetLevelError) {
    logger::setLevel(logger::Level::ERR);
    logger::error("Test message: {}", "success");
    logger::warning("This should not be printed: {}", 42);

    test_msg += "[ERROR]: Test message: success\n";
}
