// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
#if !defined(__LIBSYCL_MAJOR_VERSION) || __LIBSYCL_MAJOR_VERSION < 0
#error "__LIBSYCL_MAJOR_VERSION is not properly defined"
#endif
#if !defined(__LIBSYCL_MINOR_VERSION) || __LIBSYCL_MINOR_VERSION < 0
#error "__LIBSYCL_MINOR_VERSION is not properly defined"
#endif
#if !defined(__LIBSYCL_PATCH_VERSION) || __LIBSYCL_PATCH_VERSION < 0
#error "__LIBSYCL_PATCH_VERSION is not properly defined"
#endif
  return 0;
}
