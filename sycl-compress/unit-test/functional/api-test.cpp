//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "sycl-compress/sycl-compress.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace sycl_compress;
class syclCompressCorrectnessTest : public ::testing::Test {};

// Simple (de)compression of a string
TEST_F(syclCompressCorrectnessTest, CompressionTest) {

  std::string input = "Hello, World!";
  size_t compressedSize = 0;
  auto compressedData = ZSTDCompressor::CompressBlob(
      input.c_str(), input.size(), compressedSize, 1);

  ASSERT_NE(compressedData, nullptr);
  ASSERT_GT(compressedSize, 0);

  size_t decompressedSize = 0;
  auto decompressedData = ZSTDCompressor::DecompressBlob(
      compressedData.get(), compressedSize, decompressedSize);

  ASSERT_NE(decompressedData, nullptr);
  ASSERT_GT(decompressedSize, 0);

  std::string decompressedStr(decompressedData.get(), decompressedSize);
  ASSERT_EQ(input, decompressedStr);
}

// Test getting error code and error string.
// Intentionally give incorrect input to decompress
// to trigger an error.
TEST_F(syclCompressCorrectnessTest, NegativeErrorTest) {
  std::string input = "Hello, World!";
  size_t decompressedSize = 0;
  auto compressedData = ZSTDCompressor::DecompressBlob(
      input.c_str(), input.size(), decompressedSize);

  int errorCode = ZSTDCompressor::GetLastError();
  ASSERT_NE(errorCode, 0);

  std::string errorString = ZSTDCompressor::GetErrorString(errorCode);
  ASSERT_NE(errorString, "No error detected");
}

// Test that the error code is 0 after a successful (de)compression.
TEST_F(syclCompressCorrectnessTest, PositiveErrorTest) {
  std::string input = "Hello, World!";
  [[maybe_unused]] size_t compressedSize = 0;
  [[maybe_unused]] auto compressedData = ZSTDCompressor::CompressBlob(
      input.c_str(), input.size(), compressedSize, 1);

  int errorCode = ZSTDCompressor::GetLastError();
  ASSERT_EQ(errorCode, 0);

  std::string errorString = ZSTDCompressor::GetErrorString(errorCode);
  ASSERT_EQ(errorString, "No error detected");
}

// Test passing empty input to (de)compress.
// There should be no error and the output should be empty.
TEST_F(syclCompressCorrectnessTest, EmptyInputTest) {
  std::string input = "";
  size_t compressedSize = 0;
  auto compressedData = ZSTDCompressor::CompressBlob(
      input.c_str(), input.size(), compressedSize, 1);

  ASSERT_NE(compressedData, nullptr);
  ASSERT_GT(compressedSize, 0);
  ASSERT_EQ(ZSTDCompressor::GetLastError(), 0);

  size_t decompressedSize = 0;
  auto decompressedData = ZSTDCompressor::DecompressBlob(
      compressedData.get(), compressedSize, decompressedSize);

  ASSERT_NE(decompressedData, nullptr);
  ASSERT_EQ(decompressedSize, 0);
  ASSERT_EQ(ZSTDCompressor::GetLastError(), 0);

  std::string decompressedStr(decompressedData.get(), decompressedSize);
  ASSERT_EQ(input, decompressedStr);
}
