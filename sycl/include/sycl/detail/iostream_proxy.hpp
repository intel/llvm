#pragma once
#include <istream>
#include <ostream>

namespace std {
#if defined(_MT) && defined(_DLL)
#define __SYCL_EXTERN_STREAM_ATTRS __declspec(dllimport)
#else
#define __SYCL_EXTERN_STREAM_ATTRS
#endif // defined(_MT) && defined(_DLL)

/// Linked to standard input
extern __SYCL_EXTERN_STREAM_ATTRS istream cin;
/// Linked to standard output
extern __SYCL_EXTERN_STREAM_ATTRS ostream cout;
/// Linked to standard error (unbuffered)
extern __SYCL_EXTERN_STREAM_ATTRS ostream cerr;
/// Linked to standard error (buffered)
extern __SYCL_EXTERN_STREAM_ATTRS ostream clog;
#undef __SYCL_EXTERN_STREAM_ATTRS
} // namespace std