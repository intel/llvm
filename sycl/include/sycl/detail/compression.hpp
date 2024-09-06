//==---------- compression.hpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>
#include <zstd.h>

#define ZSTD_CONTENTSIZE_UNKNOWN (0ULL - 1)
#define ZSTD_CONTENTSIZE_ERROR (0ULL - 2)

namespace sycl {
inline namespace _V1 {
namespace detail {

// Singleton class to handle ZSTD compression and decompression.
class ZSTDCompressor {
private:
  // Initialize ZSTD context and error code.
  ZSTDCompressor() {
    m_ZSTD_compression_ctx = static_cast<void *>(ZSTD_createCCtx());
    m_ZSTD_decompression_ctx = static_cast<void *>(ZSTD_createDCtx());

    if (!m_ZSTD_compression_ctx || !m_ZSTD_decompression_ctx) {
      std::cerr << "Error creating ZSTD contexts. \n";
    }

    m_lastError = 0;
  }

  // Free ZSTD contexts.
  ~ZSTDCompressor() {
    ZSTD_freeCCtx(static_cast<ZSTD_CCtx *>(m_ZSTD_compression_ctx));
    ZSTD_freeDCtx(static_cast<ZSTD_DCtx *>(m_ZSTD_decompression_ctx));
  }

  ZSTDCompressor(const ZSTDCompressor &) = delete;
  ZSTDCompressor &operator=(const ZSTDCompressor &) = delete;

  // Get the singleton instance of the ZSTDCompressor class.
  static ZSTDCompressor &GetSingletonInstance() {
    static ZSTDCompressor instance;
    return instance;
  }

  // Public APIs
public:
  // Return 0 is last (de)compression was successful, otherwise return error
  // code.
  static int GetLastError() { return GetSingletonInstance().m_lastError; }

  // Returns a string representation of the error code.
  // If the error code is 0, it returns "No error detected".
  static std::string GetErrorString(int code) {
    return ZSTD_getErrorName(code);
  }

  // Blob (de)compression do not assume format/structure of the input buffer.
  static std::unique_ptr<char> CompressBlob(const char *src, size_t srcSize,
                                            size_t &dstSize, int level) {
    auto &instance = GetSingletonInstance();

    // Get maximum size of the compressed buffer and allocate it.
    auto dstBufferSize = ZSTD_compressBound(srcSize);
    auto dstBuffer = std::unique_ptr<char>(new char[dstBufferSize]);

    // Compress the input buffer.
    dstSize = ZSTD_compressCCtx(
        static_cast<ZSTD_CCtx *>(instance.m_ZSTD_compression_ctx),
        static_cast<void *>(dstBuffer.get()), dstBufferSize,
        static_cast<const void *>(src), srcSize, level);

    // Store the error code if compression failed.
    if (ZSTD_isError(dstSize))
      instance.m_lastError = dstSize;
    else
      instance.m_lastError = 0;

    // Pass ownership of the buffer to the caller.
    return std::move(dstBuffer);
  }

  static std::unique_ptr<unsigned char>
  DecompressBlob(const char *src, size_t srcSize, size_t &dstSize) {
    auto &instance = GetSingletonInstance();

    // Size of decompressed image can be larger than what we can allocate
    // on heap. In that case, we need to use streaming decompression.
    // TODO: Throw if the decompression size is too large.
    auto dstBufferSize = ZSTD_getFrameContentSize(src, srcSize);

    if (dstBufferSize == ZSTD_CONTENTSIZE_UNKNOWN ||
        dstBufferSize == ZSTD_CONTENTSIZE_ERROR) {

      std::cerr << "Error determining size of uncompressed data\n";
      dstSize = 0;
      instance.m_lastError = dstBufferSize;
      return nullptr;
    }

    // Allocate buffer for decompressed data.
    auto dstBuffer =
        std::unique_ptr<unsigned char>(new unsigned char[dstBufferSize]);

    dstSize = ZSTD_decompressDCtx(
        static_cast<ZSTD_DCtx *>(instance.m_ZSTD_decompression_ctx),
        static_cast<void *>(dstBuffer.get()), dstBufferSize,
        static_cast<const void *>(src), srcSize);

    // In case of decompression error, return the error message and set dstSize
    // to 0.
    if (ZSTD_isError(dstSize)) {
      instance.m_lastError = dstSize;
      dstSize = 0;
    }

    // Pass ownership of the buffer to the caller.
    return std::move(dstBuffer);
  }

  // Data fields
private:
  int m_lastError;
  // ZSTD context. Reusing ZSTD context speeds up subsequent (de)compression.
  // Storing as void* to avoid including ZSTD headers in this file.
  void *m_ZSTD_compression_ctx;
  void *m_ZSTD_decompression_ctx;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
