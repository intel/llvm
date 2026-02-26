// Copyright (C) 2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <ur_print.hpp>

using urGraphDumpContentsExpTest = uur::urGraphPopulatedExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphDumpContentsExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphDumpContentsExpTest, Success) {
  auto tempDir = std::filesystem::temp_directory_path();
  auto filePath = tempDir / "test_graph_dump.dot";
  std::cerr << "OUT FILE PATH: " << filePath << std::endl;

  ur_result_t result = urGraphDumpContentsExp(graph, filePath.string().c_str());
  std::cerr << "urGraphDumpContentsExp result: " << result << std::endl;
  ASSERT_SUCCESS(result);

  // Check if file exists
  bool exists = std::filesystem::exists(filePath);
  std::cerr << "File exists: " << exists << std::endl;

  if (exists) {
    auto fileSize = std::filesystem::file_size(filePath);
    std::cerr << "File size: " << fileSize << " bytes" << std::endl;

    auto status = std::filesystem::status(filePath);
    std::cerr << "File type: " << static_cast<int>(status.type()) << std::endl;
    std::cerr << "File permissions: " << std::oct
              << static_cast<int>(status.permissions()) << std::dec
              << std::endl;
  }

  std::ifstream file(filePath);
  std::cerr << "file.good(): " << file.good() << std::endl;
  std::cerr << "file.is_open(): " << file.is_open() << std::endl;
  std::cerr << "file.fail(): " << file.fail() << std::endl;
  std::cerr << "file.bad(): " << file.bad() << std::endl;
  std::cerr << "file.eof(): " << file.eof() << std::endl;

  ASSERT_TRUE(file.good());

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  std::cerr << "Content length: " << content.length() << std::endl;
  std::cerr << "GRAPH DUMP:\n" << content << std::endl;
  ASSERT_NE(content.find("digraph"), std::string::npos);

  file.close();
  std::filesystem::remove(filePath);
}

TEST_P(urGraphDumpContentsExpTest, SuccessEmptyGraph) {
  auto tempDir = std::filesystem::temp_directory_path();
  auto filePath = tempDir / "test_empty_graph_dump.dot";
  std::cerr << "OUT FILE PATH: " << filePath << std::endl;

  ur_exp_graph_handle_t emptyGraph = nullptr;
  ASSERT_SUCCESS(urGraphCreateExp(context, &emptyGraph));

  ur_result_t result =
      urGraphDumpContentsExp(emptyGraph, filePath.string().c_str());
  std::cerr << "urGraphDumpContentsExp result: " << result << std::endl;
  ASSERT_SUCCESS(result);

  // Check if file exists
  bool exists = std::filesystem::exists(filePath);
  std::cerr << "File exists: " << exists << std::endl;

  if (exists) {
    auto fileSize = std::filesystem::file_size(filePath);
    std::cerr << "File size: " << fileSize << " bytes" << std::endl;

    auto status = std::filesystem::status(filePath);
    std::cerr << "File type: " << static_cast<int>(status.type()) << std::endl;
    std::cerr << "File permissions: " << std::oct
              << static_cast<int>(status.permissions()) << std::dec
              << std::endl;
  }

  std::ifstream file(filePath);
  std::cerr << "file.good(): " << file.good() << std::endl;
  std::cerr << "file.is_open(): " << file.is_open() << std::endl;
  std::cerr << "file.fail(): " << file.fail() << std::endl;
  std::cerr << "file.bad(): " << file.bad() << std::endl;
  std::cerr << "file.eof(): " << file.eof() << std::endl;

  ASSERT_TRUE(file.good());

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  std::cerr << "Content length: " << content.length() << std::endl;
  std::cerr << "GRAPH DUMP:\n" << content << std::endl;
  ASSERT_NE(content.find("digraph"), std::string::npos);

  file.close();
  std::filesystem::remove(filePath);
  ASSERT_SUCCESS(urGraphDestroyExp(emptyGraph));
}

TEST_P(urGraphDumpContentsExpTest, InvalidNullHandleGraph) {
  const char *filePath = "test_graph_dump.dot";
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphDumpContentsExp(nullptr, filePath));
}

TEST_P(urGraphDumpContentsExpTest, InvalidNullPointerFilePath) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphDumpContentsExp(graph, nullptr));
}

TEST_P(urGraphDumpContentsExpTest, InvalidFilePath) {
  const char *invalidPath = "/invalid/path/that/does/not/exist/graph.dot";
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNKNOWN,
                   urGraphDumpContentsExp(graph, invalidPath));
}
