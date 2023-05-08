// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_ADAPTER_REG_TEST_HELPERS_H
#define UR_ADAPTER_REG_TEST_HELPERS_H

#include "ur_adapter_registry.hpp"
#include "ur_util.hpp"
#include <functional>
#include <gtest/gtest.h>

struct adapterRegSearchTest : ::testing::Test {
    ur_loader::AdapterRegistry registry;

    const fs::path testLibName =
        MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0");
    std::function<bool(const fs::path &)> isTestLibName =
        [this](const fs::path &path) { return path == testLibName; };

    std::function<bool(const std::vector<fs::path> &)> hasTestLibName =
        [this](const std::vector<fs::path> &paths) {
            return std::any_of(paths.cbegin(), paths.cend(), isTestLibName);
        };

    fs::path testAdapterEnvPath;
    std::function<bool(const fs::path &)> isTestEnvPath =
        [this](const fs::path &path) {
            if (testAdapterEnvPath.empty()) {
                return false;
            }

            auto fullPath = testAdapterEnvPath / testLibName;
            return path == fullPath;
        };

    std::function<bool(const std::vector<fs::path> &)> hasTestEnvPath =
        [this](const std::vector<fs::path> &paths) {
            return std::any_of(paths.cbegin(), paths.cend(), isTestEnvPath);
        };

    fs::path testCurPath;
    std::function<bool(const fs::path &)> isCurPath =
        [this](const fs::path &path) {
            if (testCurPath.empty()) {
                return false;
            }

            auto fullPath = testCurPath / testLibName;
            return path == fullPath;
        };

    std::function<bool(const std::vector<fs::path> &)> hasCurPath =
        [this](const std::vector<fs::path> &paths) {
            return std::any_of(paths.cbegin(), paths.cend(), isCurPath);
        };

    void SetUp() override {
        try {
            auto testSearchPathsEnvOpt =
                getenv_to_vec("TEST_ADAPTER_SEARCH_PATH");
            if (testSearchPathsEnvOpt.has_value()) {
                testAdapterEnvPath =
                    fs::path(testSearchPathsEnvOpt.value().front());
            }

            auto testSearchCurPathOpt = getenv_to_vec("TEST_CUR_SEARCH_PATH");
            if (testSearchCurPathOpt.has_value()) {
                testCurPath = fs::path(testSearchCurPathOpt.value().front());
            }
        } catch (const std::invalid_argument &e) {
            FAIL() << e.what();
        }
    }
};

#endif // UR_ADAPTER_REG_TEST_HELPERS_H
