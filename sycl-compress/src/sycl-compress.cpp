#include <cstring>
#include <iostream>

#include <sycl-compress/sycl-compress.h>
#include <zstd.h>

#define ZSTD_CONTENTSIZE_UNKNOWN (0ULL - 1)
#define ZSTD_CONTENTSIZE_ERROR (0ULL - 2)

#if defined(_MSC_VER)
__declspec(dllexport)
#endif
char *
compressBlob(const char *src, size_t srcSize, size_t &dstSize, int level) {
  auto dstBufferSize = ZSTD_compressBound(srcSize);
  char *dstBuffer = static_cast<char *>(malloc(dstBufferSize));
  dstSize = ZSTD_compress(static_cast<void *>(dstBuffer), dstBufferSize,
                          static_cast<const void *>(src), srcSize, level);

  // In case of compression error, return the error message and set dstSize to
  // 0.
  if (ZSTD_isError(dstSize)) {
    std::cerr << "Error: " << ZSTD_getErrorName(dstSize) << "\n";
    strncpy(dstBuffer, ZSTD_getErrorName(dstSize), dstBufferSize);
    dstSize = 0;
  }

  return dstBuffer;
}

#if defined(_MSC_VER)
__declspec(dllexport)
#endif
char *
decompressBlob(const char *src, size_t srcSize, size_t &dstSize) {
  // Size of decompressed image can be larger than what we can allocate
  // on heap. In that case, we need to use streaming decompression.
  // TODO: Throw if the decompression size is too large.
  auto dstBufferSize = ZSTD_getFrameContentSize(src, srcSize);

  if (dstBufferSize == ZSTD_CONTENTSIZE_UNKNOWN ||
      dstBufferSize == ZSTD_CONTENTSIZE_ERROR) {
    std::cerr << "Error determining size of uncompressed data\n";
    std::cerr << "Error: " << ZSTD_getErrorName(dstBufferSize) << "\n";
    dstSize = 0;
    return nullptr;
  }

  char *dstBuffer = static_cast<char *>(malloc(dstBufferSize));
  dstSize = ZSTD_decompress(static_cast<void *>(dstBuffer), dstBufferSize,
                            static_cast<const void *>(src), srcSize);

  // In case of decompression error, return the error message and set dstSize to
  // 0.
  if (ZSTD_isError(dstSize)) {
    std::cerr << "Error: " << ZSTD_getErrorName(dstSize) << "\n";
    strncpy(dstBuffer, ZSTD_getErrorName(dstSize), dstBufferSize);
    dstSize = 0;
  }

  return dstBuffer;
}
