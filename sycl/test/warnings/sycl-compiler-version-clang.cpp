// RUN: %clangxx -fsycl %s -fsyntax-only -Xclang -verify

#include <sycl/sycl.hpp>

// expected-warning@+1 {{__SYCL_COMPILER_VERSION is deprecated, use __LIBSYCL_TIMESTAMP instead}}
#if __SYCL_COMPILER_VERSION >= 2024
#endif
