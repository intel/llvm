// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#if SYCL_BACKEND_OPENCL == 1
constexpr bool backend_opencl_macro_defined = true;
#else
constexpr bool backend_opencl_macro_defined = false;
#endif

#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK == 1
constexpr bool sub_group_mask_macro_defined = true;
#else
constexpr bool sub_group_mask_macro_defined = false;
#endif

#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO == 3
constexpr bool backend_level_zero_macro_defined = true;
#else
constexpr bool backend_level_zero_macro_defined = false;
#endif

int main() {
  static_assert(backend_opencl_macro_defined);
  static_assert(sub_group_mask_macro_defined);
  static_assert(backend_level_zero_macro_defined);

  return 0;
}
