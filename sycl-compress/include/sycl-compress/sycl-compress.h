//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include <memory>

namespace sycl_compress {

// Singleton class to handle ZSTD compression and decompression.
class
#ifdef _WIN32
#ifdef SYCL_COMPRESS_BUILD
    __declspec(dllexport) // When building sycl-compress
#else
    __declspec(dllimport) // When using sycl-compress headers in dependencies.
#endif
#endif
    ZSTDCompressor {
private:
  ZSTDCompressor();
  ~ZSTDCompressor();

  ZSTDCompressor(const ZSTDCompressor &) = delete;
  ZSTDCompressor &operator=(const ZSTDCompressor &) = delete;

  // Get the singleton instance of the ZSTDCompressor class.
  static ZSTDCompressor &GetSingletonInstance();

  // Public APIs
public:
  // Return 0 is last (de)compression was successful, otherwise return error
  // code.
  static int GetLastError();

  // Returns a string representation of the error code.
  // If the eror code is 0, it returns an empty string.
  static std::string GetErrorString(int code);

  // Blob (de)compression do not assume format/structure of the input buffer.
  static std::unique_ptr<char> CompressBlob(const char *src, size_t srcSize,
                                            size_t &dstSize, int level);

  static std::unique_ptr<char> DecompressBlob(const char *src, size_t srcSize,
                                              size_t &dstSize);

  // Data fields
private:
  int m_lastError;
  // ZSTD context. Reusing ZSTD context speeds up subsequent (de)compression.
  // Storing as void* to avoid including ZSTD headers in this file.
  void *m_ZSTD_compression_ctx;
  void *m_ZSTD_decompression_ctx;
};
} // namespace sycl_compress