// Copyright (C) 2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <ur_util.hpp>

using urGraphDumpContentsExpTest = uur::urGraphPopulatedExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphDumpContentsExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// Lit runs in parallel, so embedding pid + thread id prevents concurrent
// read / write to the same file.
std::string getUniqueTestFilename(const std::string &base_name) {
  std::stringstream ss;
  ss << base_name << "_" << ur_getpid() << "_" << std::this_thread::get_id()
     << ".dot";
  return ss.str();
}

TEST_P(urGraphDumpContentsExpTest, Success) {
  std::string filename = getUniqueTestFilename("graph_dump");

  ASSERT_SUCCESS(urGraphDumpContentsExp(graph, filename.c_str()));

  std::ifstream file(filename);
  ASSERT_TRUE(file.good());

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  ASSERT_NE(content.find("digraph"), std::string::npos);

  file.close();
  std::filesystem::remove(filename);
}

TEST_P(urGraphDumpContentsExpTest, SuccessEmptyGraph) {
  std::string filename = getUniqueTestFilename("empty_graph_dump");

  ur_exp_graph_handle_t emptyGraph = nullptr;
  ASSERT_SUCCESS(urGraphCreateExp(context, &emptyGraph));

  ASSERT_SUCCESS(urGraphDumpContentsExp(emptyGraph, filename.c_str()));

  std::ifstream file(filename);
  ASSERT_TRUE(file.good());

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  ASSERT_NE(content.find("digraph"), std::string::npos);

  file.close();
  std::filesystem::remove(filename);
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
