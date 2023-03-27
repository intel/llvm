// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UNIT_LOGGER_TEST_FIXTURES_HPP
#define UR_UNIT_LOGGER_TEST_FIXTURES_HPP

#include <filesystem>
#include <gtest/gtest.h>

#include "helpers.h"
#include "logger/ur_logger.hpp"

class LoggerFromEnvVar : public ::testing::Test {
  protected:
    int ret = -1;

    void SetUp() override {
        std::string env_var_value = "level:info;output:stderr";
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

class LoggerWithFileSink : public ::testing::Test {
  protected:
    std::filesystem::path file_name = "ur_test_logger.log";
    std::filesystem::path file_path = file_name;
    std::string logger_name = "test";
    std::string test_msg = "<" + logger_name + ">";
    std::unique_ptr<logger::Logger> logger;

    void TearDown() override {
        logger.reset();

        auto test_log = std::ifstream(file_path);
        ASSERT_TRUE(test_log.good());
        std::stringstream printed_msg;
        printed_msg << test_log.rdbuf();
        test_log.close();

        ASSERT_GT(std::filesystem::remove_all(*file_path.begin()), 0);
        ASSERT_EQ(printed_msg.str(), test_msg);
    }
};

class DefaultLoggerWithFileSink : public LoggerWithFileSink {
  protected:
    void SetUp() override {
        logger = std::make_unique<logger::Logger>(
            logger::Level::WARN,
            std::make_unique<logger::FileSink>(logger_name, file_path));
    }
};

#endif // UR_UNIT_LOGGER_TEST_FIXTURES_HPP
