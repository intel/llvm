#include <iostream>
#include <zstd.h>

__attribute__((visibility("default"))) int compressBlob(void *src, size_t srcSize,
                                                        void *dst, int level) {
    void* dstBuffer = malloc(srcSize);
    size_t dstSize = ZSTD_compress(src, srcSize, dstBuffer, srcSize, level);
    dst = dstBuffer;
    return dstSize;
}

int main() {
    return 0;
}