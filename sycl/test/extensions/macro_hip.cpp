// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl -fsyntax-only %s
// REQUIRES: hip

#include <sycl/sycl.hpp>

#if SYCL_EXT_ONEAPI_BACKEND_HIP == 1
constexpr bool macro_defined = true;
#else
constexpr bool macro_defined = false;
#endif

int main() {
  static_assert(macro_defined);

  return 0;
}
