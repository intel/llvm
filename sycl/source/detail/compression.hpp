//==---------- compression.hpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#ifndef SYCL_RT_ZSTD_NOT_AVAIABLE

#include <sycl/exception.hpp>

#include <iostream>
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
  ZSTDCompressor() {}

  ZSTDCompressor(const ZSTDCompressor &) = delete;
  ZSTDCompressor &operator=(const ZSTDCompressor &) = delete;
  ~ZSTDCompressor() {}

  // Get the singleton instance of the ZSTDCompressor class.
  static ZSTDCompressor &GetSingletonInstance() {
    static ZSTDCompressor instance;
    return instance;
  }

  // Public APIs
public:
  // Blob (de)compression do not assume format/structure of the input buffer.
  // This function can be used in future for compression in on-disk cache.
  static std::unique_ptr<char> CompressBlob(const char *src, size_t srcSize,
                                            size_t &dstSize, int level) {
    auto &instance = GetSingletonInstance();

    // Lazy initialize compression context.
    if (!instance.m_ZSTD_compression_ctx) {

      // Call ZSTD_createCCtx() and ZSTD_freeCCtx() to create and free the
      // context.
      instance.m_ZSTD_compression_ctx =
          std::unique_ptr<ZSTD_CCtx, size_t (*)(ZSTD_CCtx *)>(ZSTD_createCCtx(),
                                                              ZSTD_freeCCtx);
      if (!instance.m_ZSTD_compression_ctx) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Failed to create ZSTD compression context");
      }
    }

    // Get maximum size of the compressed buffer and allocate it.
    auto dstBufferSize = ZSTD_compressBound(srcSize);
    auto dstBuffer = std::unique_ptr<char>(new char[dstBufferSize]);

    if (!dstBuffer)
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Failed to allocate memory for compressed data");

    // Compress the input buffer.
    dstSize =
        ZSTD_compressCCtx(instance.m_ZSTD_compression_ctx.get(),
                          static_cast<void *>(dstBuffer.get()), dstBufferSize,
                          static_cast<const void *>(src), srcSize, level);

    // Store the error code if compression failed.
    if (ZSTD_isError(dstSize))
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            ZSTD_getErrorName(dstSize));

    // Pass ownership of the buffer to the caller.
    return dstBuffer;
  }

  static size_t GetDecompressedSize(const char *src, size_t srcSize) {
    size_t dstBufferSize = ZSTD_getFrameContentSize(src, srcSize);

    if (dstBufferSize == ZSTD_CONTENTSIZE_UNKNOWN ||
        dstBufferSize == ZSTD_CONTENTSIZE_ERROR) {
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Error determining size of uncompressed data.");
    }
    return dstBufferSize;
  }

  static std::unique_ptr<char> DecompressBlob(const char *src, size_t srcSize,
                                              size_t &dstSize) {
    auto &instance = GetSingletonInstance();

    // Lazy initialize decompression context.
    if (!instance.m_ZSTD_decompression_ctx) {

      // Call ZSTD_createDCtx() and ZSTD_freeDCtx() to create and free the
      // context.
      instance.m_ZSTD_decompression_ctx =
          std::unique_ptr<ZSTD_DCtx, size_t (*)(ZSTD_DCtx *)>(ZSTD_createDCtx(),
                                                              ZSTD_freeDCtx);
      if (!instance.m_ZSTD_decompression_ctx) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Failed to create ZSTD decompression context");
      }
    }

    // Size of decompressed image can be larger than what we can allocate
    // on heap. In that case, we need to use streaming decompression.
    auto dstBufferSize = GetDecompressedSize(src, srcSize);

    // Allocate buffer for decompressed data.
    auto dstBuffer = std::unique_ptr<char>(new char[dstBufferSize]);

    if (!dstBuffer)
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Failed to allocate memory for decompressed data");

    dstSize =
        ZSTD_decompressDCtx(instance.m_ZSTD_decompression_ctx.get(),
                            static_cast<void *>(dstBuffer.get()), dstBufferSize,
                            static_cast<const void *>(src), srcSize);

    // In case of decompression error, return the error message and set dstSize
    // to 0.
    if (ZSTD_isError(dstSize)) {
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            ZSTD_getErrorName(dstSize));
    }

    // Pass ownership of the buffer to the caller.
    return dstBuffer;
  }

  // Data fields
private:
  // ZSTD contexts. Reusing ZSTD context speeds up subsequent (de)compression.
  std::unique_ptr<ZSTD_CCtx, size_t (*)(ZSTD_CCtx *)> m_ZSTD_compression_ctx{
      nullptr, nullptr};
  std::unique_ptr<ZSTD_DCtx, size_t (*)(ZSTD_DCtx *)> m_ZSTD_decompression_ctx{
      nullptr, nullptr};
};
} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // SYCL_RT_ZSTD_NOT_AVAIABLE
