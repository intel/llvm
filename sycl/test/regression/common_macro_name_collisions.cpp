// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests that common macro names do not cause problems in the DPC++ headers.
// More common macro names can be added in the future.

#define VL
#define DIM
#define DIMS
#define NDIMS

#include <sycl/sycl.hpp>
