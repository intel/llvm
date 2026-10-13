// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: UR_ADAPTERS_SEARCH_PATH="" prefilter-test

#include "fixtures.hpp"

#ifndef _WIN32

TEST_F(adapterPreFilterTest, testPrefilterAcceptFilterSingleBackend) {
  SetUp("level_zero:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_FALSE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterAcceptFilterMultipleBackends) {
  SetUp("level_zero:*;opencl:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterDiscardFilterSingleBackend) {
  SetUp("!level_zero:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_FALSE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_TRUE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterDiscardFilterMultipleBackends) {
  SetUp("!level_zero:*;!cuda:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_FALSE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterAcceptAndDiscardFilter) {
  SetUp("level_zero:*;!cuda:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_FALSE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterDiscardFilterAll) {
  SetUp("*:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_TRUE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterWithInvalidMissingBackend) {
  SetUp(":garbage");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_TRUE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterWithInvalidBackend) {
  SetUp("garbage:0");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_TRUE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_TRUE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_TRUE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterWithNotAllAndAcceptFilter) {
  SetUp("level_zero:*;!*:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_FALSE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_FALSE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testPrefilterWithNotAllFilter) {
  SetUp("!*:*");
  auto levelZeroExists =
      std::any_of(registry->cbegin(), registry->cend(), haslevelzeroLibName);
  EXPECT_FALSE(levelZeroExists);
  auto openclExists =
      std::any_of(registry->cbegin(), registry->cend(), hasOpenclLibName);
  EXPECT_FALSE(openclExists);
  auto cudaExists =
      std::any_of(registry->cbegin(), registry->cend(), hasCudaLibName);
  EXPECT_FALSE(cudaExists);
}

TEST_F(adapterPreFilterTest, testStaticAdapterFilteredOutByBackend) {
  SetUp("level_zero:gpu");
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_level_zero"));
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_level_zero_v2"));
  EXPECT_FALSE(registry->includesAdapter("ur_adapter_opencl"));
}

TEST_F(adapterPreFilterTest, testStaticAdapterIncludedWhenSelected) {
  SetUp("opencl:gpu");
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_opencl"));
  EXPECT_FALSE(registry->includesAdapter("ur_adapter_level_zero"));
  EXPECT_FALSE(registry->includesAdapter("ur_adapter_level_zero_v2"));
}

TEST_F(adapterPreFilterTest, testStaticAdapterExcludedByNegativeFilter) {
  SetUp("!opencl:*");
  EXPECT_FALSE(registry->includesAdapter("ur_adapter_opencl"));
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_level_zero"));
}

TEST_F(adapterPreFilterTest, testStaticAdapterIncludedWhenAcceptAll) {
  SetUp("*:*");
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_level_zero"));
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_level_zero_v2"));
  EXPECT_TRUE(registry->includesAdapter("ur_adapter_opencl"));
}

#endif
