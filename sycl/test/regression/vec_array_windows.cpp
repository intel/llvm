// Test to isolate sycl::vec regression after
// https://github.com/intel/llvm/pull/14130. This PR caused sycl::vec to use
// std::array as its underlying storage. However, operations on std::array
// may emit debug-mode-only functions, on which the device compiler may fail.

// REQUIRES: windows

// RUN: %clangxx -fsycl -fpreview-breaking-changes -D_DEBUG -fsycl-device-only %s

#include <sycl/sycl.hpp>

SYCL_EXTERNAL auto GetFirstElement(sycl::vec<int, 3> v) { return v[0]; }
