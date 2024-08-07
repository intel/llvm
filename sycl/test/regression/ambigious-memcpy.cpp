// detail::memcpy, even though in a different namespace, can cause ambiguity
// with libc's memcpy, due to argument dependent lookup (ADL). This compile-only
// test checks for ambigiouity in call to detail::memcpy.

// RUN: %clang -fsycl -fsycl-device-only -fsyntax-only %s

#include <sycl/vector.hpp>

template <typename T> void foo(T *dst, T *src, size_t count) {
  memcpy(dst, src, count * sizeof(T));
}

using T = sycl::vec<int, 1>;

SYCL_EXTERNAL void bar(T *dst, T *src, size_t count) {
  foo(dst, src, count * sizeof(T));
}
