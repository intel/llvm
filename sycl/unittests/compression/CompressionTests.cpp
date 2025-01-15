//==------- CompressionTests.cpp --- compression unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/compression.hpp>

#include <string>

#include <gtest/gtest.h>

using namespace sycl::detail;

TEST(CompressionTest, SimpleCompression) {

  // Data to compress.
  std::string data = "Hello World! I'm about to get compressed :P";
  size_t compressedDataSize = 0;

  auto compressedData = ZSTDCompressor::CompressBlob(
      data.c_str(), data.size(), compressedDataSize, /*Compression level*/ 3);

  // Check if compression was successful.
  EXPECT_NE(compressedData, nullptr);
  EXPECT_GT(compressedDataSize, (size_t)0);

  // Decompress the data.
  size_t decompressedSize = 0;
  auto decompressedData = ZSTDCompressor::DecompressBlob(
      compressedData.get(), compressedDataSize, decompressedSize);

  ASSERT_NE(decompressedData, nullptr);
  ASSERT_GT(decompressedSize, (size_t)0);

  // Check if decompressed data is same as original data.
  std::string decompressedStr((char *)decompressedData.get(), decompressedSize);
  ASSERT_EQ(data, decompressedStr);
}

// Test getting error code and error string.
// Intentionally give incorrect input to decompress
// to trigger an error.
TEST(CompressionTest, NegativeErrorTest) {
  std::string input = "Hello, World!";
  size_t decompressedSize = 0;
  bool threwException = false;
  try {
    auto compressedData = ZSTDCompressor::DecompressBlob(
        input.c_str(), input.size(), decompressedSize);
  } catch (...) {
    threwException = true;
  }

  ASSERT_TRUE(threwException);
}

// Test passing empty input to (de)compress.
// There should be no error and the output should be empty.
TEST(CompressionTest, EmptyInputTest) {
  std::string input = "";
  size_t compressedSize = 0;
  auto compressedData = ZSTDCompressor::CompressBlob(
      input.c_str(), input.size(), compressedSize, 1);

  ASSERT_NE(compressedData, nullptr);
  ASSERT_GT(compressedSize, (size_t)0);

  size_t decompressedSize = 0;
  auto decompressedData = ZSTDCompressor::DecompressBlob(
      compressedData.get(), compressedSize, decompressedSize);

  ASSERT_NE(decompressedData, nullptr);
  ASSERT_EQ(decompressedSize, (size_t)0);

  std::string decompressedStr((char *)decompressedData.get(), decompressedSize);
  ASSERT_EQ(input, decompressedStr);
}
