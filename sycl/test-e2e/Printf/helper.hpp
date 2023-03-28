#if defined(__SYCL_DEVICE_ONLY__) && defined(TEST_CONSTANT_AS)
// On device side, we have to put format string into a constant address space
// FIXME: remove this header completely once the toolchain's support for
// generic address-spaced format strings is stable.
#define FORMAT_STRING(X) static const __attribute__((opencl_constant)) char X[]
#else
#define FORMAT_STRING(X) static const char X[]
#endif
