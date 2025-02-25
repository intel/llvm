// Test to isolate sycl::vec regression after
// https://github.com/intel/llvm/pull/14130. This PR caused sycl::vec to use
// std::array as its underlying storage. However, operations on std::array
// may emit debug-mode-only functions, on which the device compiler may fail.

// REQUIRES: windows

// RUN: %clangxx -fsycl -D_DEBUG %s -fsycl-device-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning

#include <sycl/sycl.hpp>

// expected-error@* {{SYCL kernel cannot call a variadic function}}
// expected-error@* {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
// expected-error@* {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
SYCL_EXTERNAL auto GetFirstElement(sycl::vec<int, 3> v) { return v[0]; }
