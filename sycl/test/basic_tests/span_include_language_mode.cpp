// RUN: %clangxx -fsycl -fsyntax-only -std=c++17 -Xclang -verify %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DCXX20 -Xclang -verify %s
// expected-no-diagnostics

// <sycl/sycl.hpp> must not pull in <span> before C++20: MSVC's STL emits
// warning STL4038 for every translation unit that does so.

#include <sycl/sycl.hpp>

#ifdef CXX20
#if __has_include(<span>) && !defined(__cpp_lib_span)
#error "<span> should be included in C++20 mode"
#endif
#endif
