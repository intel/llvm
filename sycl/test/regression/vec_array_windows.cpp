// Test to isolate sycl::vec regression after
// https://github.com/intel/llvm/pull/14130. This PR caused sycl::vec to use
// std::array as its underlying storage. However, operations on std::array
// may emit debug-mode-only functions, on which the device compiler may fail.

// REQUIRES: windows

// RUN: not %clangxx -fsycl -D_DEBUG -I %sycl_include %s -fsycl-device-only 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: error: SYCL kernel cannot call a variadic function
SYCL_EXTERNAL auto GetFirstElement(sycl::vec<int, 3> v) { return v[0]; }
