// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Check that the experimental CUDA interop header doesn't have any warnings.
// This is a special test because this header requires a specific macro to be
// set when it is included.

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
