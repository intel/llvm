// Test to isolate sycl::vec regression after
// https://github.com/intel/llvm/pull/14130. This PR caused sycl::vec to use
// std::array as its underlying storage. However, operations on std::array
// may emit debug-mode-only functions, on which the device compiler may fail.

// REQUIRES: windows

// RUN: %clangxx -fsycl -D_DEBUG %s -fsycl-device-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes -D_DEBUG -fsycl-device-only %s %}

#include <sycl/sycl.hpp>

// expected-no-diagnostics
//
// Our current implementation automatically opts-in for a new implementation if
// that is possible without breaking ABI.
// However, depending on the environment (used STL implementation, in
// particular) it may not be the case. Therefore, the lines below are kept for
// reference of how an error would look like in a problematic environment.
// not-expected-error@* {{SYCL kernel cannot call a variadic function}}
// not-expected-error@* {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
// not-expected-error@* {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
SYCL_EXTERNAL auto GetFirstElement(sycl::vec<int, 3> v) { return v[0]; }
