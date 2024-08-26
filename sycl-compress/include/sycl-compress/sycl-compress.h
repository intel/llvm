#if defined(_MSC_VER)
__declspec(dllexport)
#endif
char *compressBlob(const char *src, size_t srcSize, size_t &dstSize, int level);


#if defined(_MSC_VER)
__declspec(dllexport)
#endif
char *decompressBlob(const char *src, size_t srcSize, size_t &dstSize);
