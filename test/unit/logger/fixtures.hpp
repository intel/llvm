// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_UNIT_LOGGER_TEST_FIXTURES_HPP
#define UR_UNIT_LOGGER_TEST_FIXTURES_HPP

#include <gtest/gtest.h>

#include "helpers.h"
#include "logger/ur_logger.hpp"

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

#endif // UR_UNIT_LOGGER_TEST_FIXTURES_HPP
