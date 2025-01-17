// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include "fixtures.hpp"
#include "ur_api.h"
#include <gtest/gtest.h>
#include <thread>
#include <ur_print.hpp>

TEST(urLoaderLifetime, ST) {
  ur_result_t status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_ERROR_UNINITIALIZED);

  status = urLoaderInit(0, nullptr);
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
  status = urLoaderInit(0, nullptr);
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
  status = urLoaderInit(0, nullptr);
  ASSERT_EQ(status, UR_RESULT_SUCCESS);

  status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
  status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
  status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
  status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_ERROR_UNINITIALIZED);
}

TEST(urLoaderLifetime, MT) {
  std::vector<std::thread> threads;

  for (int th = 0; th < 20; ++th) {
    threads.emplace_back([] {
      for (int i = 0; i < 1000; ++i) {
        ur_result_t status = urLoaderInit(0, nullptr);
        ASSERT_EQ(status, UR_RESULT_SUCCESS);
        status = urLoaderTearDown();
        ASSERT_EQ(status, UR_RESULT_SUCCESS);
        // doing extranous urLoaderTearDown's is not legal
        // in multi-threaded contexts because it may
        // race with another thread's urLoaderInit.
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  ur_result_t status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_ERROR_UNINITIALIZED);

  status = urLoaderInit(0, nullptr);
  ASSERT_EQ(status, UR_RESULT_SUCCESS);

  status = urLoaderTearDown();
  ASSERT_EQ(status, UR_RESULT_SUCCESS);
}
