// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_ADAPTER_REG_TEST_HELPERS_H
#define UR_ADAPTER_REG_TEST_HELPERS_H

#include "ur_adapter_registry.hpp"
#include "ur_util.hpp"
#include <functional>
#include <gtest/gtest.h>

struct adapterRegSearchTest : ::testing::Test {
  ur_loader::AdapterRegistry registry;

  const fs::path testLibName = MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0");
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
      auto testSearchPathsEnvOpt = getenv_to_vec("TEST_ADAPTER_SEARCH_PATH");
      if (testSearchPathsEnvOpt.has_value()) {
        testAdapterEnvPath = fs::path(testSearchPathsEnvOpt.value().front());
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
#ifndef _WIN32
struct adapterPreFilterTest : ::testing::Test {
  ur_loader::AdapterRegistry *registry;
  const fs::path levelzeroLibName =
      MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0");
  std::function<bool(const fs::path &)> islevelzeroLibName =
      [this](const fs::path &path) { return path == levelzeroLibName; };

  std::function<bool(const std::vector<fs::path> &)> haslevelzeroLibName =
      [this](const std::vector<fs::path> &paths) {
        return std::any_of(paths.cbegin(), paths.cend(), islevelzeroLibName);
      };

  const fs::path openclLibName = MAKE_LIBRARY_NAME("ur_adapter_opencl", "0");
  std::function<bool(const fs::path &)> isOpenclLibName =
      [this](const fs::path &path) { return path == openclLibName; };

  std::function<bool(const std::vector<fs::path> &)> hasOpenclLibName =
      [this](const std::vector<fs::path> &paths) {
        return std::any_of(paths.cbegin(), paths.cend(), isOpenclLibName);
      };

  const fs::path cudaLibName = MAKE_LIBRARY_NAME("ur_adapter_cuda", "0");
  std::function<bool(const fs::path &)> isCudaLibName =
      [this](const fs::path &path) { return path == cudaLibName; };

  std::function<bool(const std::vector<fs::path> &)> hasCudaLibName =
      [this](const std::vector<fs::path> &paths) {
        return std::any_of(paths.cbegin(), paths.cend(), isCudaLibName);
      };

  void SetUp(std::string filter) {
    try {
      setenv("ONEAPI_DEVICE_SELECTOR", filter.c_str(), 1);
      registry = new ur_loader::AdapterRegistry;
    } catch (const std::invalid_argument &e) {
      FAIL() << e.what();
    }
  }
  void SetUp() override {}
  void TearDown() override { delete registry; }
};
#endif

#endif // UR_ADAPTER_REG_TEST_HELPERS_H
