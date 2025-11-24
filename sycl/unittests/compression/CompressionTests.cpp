//==------- CompressionTests.cpp --- compression unit test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../thread_safety/ThreadUtils.h"
#include <detail/compression.hpp>
#include <detail/device_binary_image.hpp>
#include <sycl/sycl.hpp>

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

// Test to check for concurrent compression and decompression.
TEST(CompressionTest, ConcurrentCompressionDecompression) {
  std::string data = "Concurrent compression and decompression test!";

  constexpr size_t ThreadCount = 20;

  Barrier b(ThreadCount);
  {
    auto testCompressDecompress = [&](size_t threadId) {
      b.wait();
      size_t compressedDataSize = 0;
      auto compressedData = ZSTDCompressor::CompressBlob(
          data.c_str(), data.size(), compressedDataSize, 3);

      ASSERT_NE(compressedData, nullptr);
      ASSERT_GT(compressedDataSize, (size_t)0);

      size_t decompressedSize = 0;
      auto decompressedData = ZSTDCompressor::DecompressBlob(
          compressedData.get(), compressedDataSize, decompressedSize);

      ASSERT_NE(decompressedData, nullptr);
      ASSERT_GT(decompressedSize, (size_t)0);

      std::string decompressedStr((char *)decompressedData.get(),
                                  decompressedSize);
      ASSERT_EQ(data, decompressedStr);
    };

    ::ThreadPool MPool(ThreadCount, testCompressDecompress);
  }
}

// Test to decompress CompressedRTDeviceImage using multiple threads.
// The idea behind this test is to ensure that a device image is
// decompressed only once even if multiple threads try to decompress
// it at the same time.
TEST(CompressionTest, ConcurrentDecompressionOfDeviceImage) {
  // Data to compress.
  std::string data = "Hello World! I'm about to get compressed :P";

  // Compress this data.
  size_t compressedSize = 0;
  auto compressedData = ZSTDCompressor::CompressBlob(data.c_str(), data.size(),
                                                     compressedSize, 1);

  unsigned char *compressedDataPtr =
      reinterpret_cast<unsigned char *>(compressedData.get());

  const char *EntryName = "Entry";
  _sycl_offload_entry_struct EntryStruct = {
      /*addr*/ nullptr, const_cast<char *>(EntryName), strlen(EntryName),
      /*flags*/ 0, /*reserved*/ 0};
  sycl_device_binary_struct BinStruct{/*Version*/ 3,
                                      /*Kind*/ 4,
                                      /*Format*/ SYCL_DEVICE_BINARY_TYPE_SPIRV,
                                      /*DeviceTargetSpec*/ nullptr,
                                      /*CompileOptions*/ nullptr,
                                      /*LinkOptions*/ nullptr,
                                      /*BinaryStart*/ compressedDataPtr,
                                      /*BinaryEnd*/ compressedDataPtr +
                                          compressedSize,
                                      /*EntriesBegin*/ &EntryStruct,
                                      /*EntriesEnd*/ &EntryStruct + 1,
                                      /*PropertySetsBegin*/ nullptr,
                                      /*PropertySetsEnd*/ nullptr};
  sycl_device_binary Bin = &BinStruct;
  CompressedRTDeviceBinaryImage Img{Bin};

  // Decompress the image with multiple threads.
  constexpr size_t ThreadCount = 20;
  Barrier b(ThreadCount);
  {
    auto testDecompress = [&](size_t threadId) {
      b.wait();
      Img.Decompress();

      // Check if decompressed data is same as original data.
      // Img.getRawData will change if there's a race in image decompression
      // and the check will fail.
      for (size_t i = 0; i < Img.getSize(); ++i) {
        ASSERT_EQ(data[i], Img.getRawData().BinaryStart[i]);
      }
    };

    ::ThreadPool MPool(ThreadCount, testDecompress);
  }
}
