// RUN: %clangxx -fsycl -fsyntax-only %s
#include <sycl/sycl.hpp>

#if !defined(SYCL_IMPLEMENTATION_ONEAPI)
#error SYCL_IMPLEMENTATION_ONEAPI is not defined
#endif

#if !defined(SYCL_FEATURE_SET_FULL)
#error SYCL_FEATURE_SET_FULL is not defined
#endif

#if !defined(SYCL_IMPLEMENTATION_INTEL)
#error SYCL_IMPLEMENTATION_INTEL is not defined
#endif
