// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl -fsyntax-only %s
// REQUIRES: esimd_emulator_be

#include <sycl/sycl.hpp>

#if SYCL_EXT_INTEL_BACKEND_ESIMD_EMULATOR == 1
constexpr bool macro_deifned = true;
#else
constexpr bool macro_deifned = false;
#endif

int main() {
  static_assert(macro_defined);

  return 0;
}
