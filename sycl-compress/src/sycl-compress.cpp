//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include <cstring>
#include <iostream>

#include <sycl-compress/sycl-compress.h>
#include <zstd.h>

#define ZSTD_CONTENTSIZE_UNKNOWN (0ULL - 1)
#define ZSTD_CONTENTSIZE_ERROR (0ULL - 2)

namespace sycl_compress {

// Singleton instance of the ZSTDCompressor class.
ZSTDCompressor &ZSTDCompressor::GetSingletonInstance() {
  static ZSTDCompressor instance;
  return instance;
}

// Initialize ZSTD context and error code.
ZSTDCompressor::ZSTDCompressor() {
  m_ZSTD_compression_ctx = static_cast<void *>(ZSTD_createCCtx());
  m_ZSTD_decompression_ctx = static_cast<void *>(ZSTD_createDCtx());

  if (!m_ZSTD_compression_ctx || !m_ZSTD_decompression_ctx) {
    std::cerr << "Error creating ZSTD contexts. \n";
  }

  m_lastError = 0;
}

// Free ZSTD contexts.
ZSTDCompressor::~ZSTDCompressor() {
  ZSTD_freeCCtx(static_cast<ZSTD_CCtx *>(m_ZSTD_compression_ctx));
  ZSTD_freeDCtx(static_cast<ZSTD_DCtx *>(m_ZSTD_decompression_ctx));
}

std::unique_ptr<char> ZSTDCompressor::CompressBlob(const char *src,
                                                   size_t srcSize,
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

std::unique_ptr<char> ZSTDCompressor::DecompressBlob(const char *src,
                                                     size_t srcSize,
                                                     size_t &dstSize) {

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
  auto dstBuffer = std::unique_ptr<char>(new char[dstBufferSize]);

  dstSize = ZSTD_decompressDCtx(
      static_cast<ZSTD_DCtx *>(instance.m_ZSTD_decompression_ctx),
      static_cast<void *>(dstBuffer.get()), dstBufferSize,
      static_cast<const void *>(src), srcSize);

  // In case of decompression error, return the error message and set dstSize to
  // 0.
  if (ZSTD_isError(dstSize)) {
    instance.m_lastError = dstSize;
    dstSize = 0;
  }

  // Pass ownership of the buffer to the caller.
  return std::move(dstBuffer);
}

int ZSTDCompressor::GetLastError() {
  return GetSingletonInstance().m_lastError;
}

std::string ZSTDCompressor::GetErrorString(int code) {
  return ZSTD_getErrorName(code);
}
} // namespace sycl_compress