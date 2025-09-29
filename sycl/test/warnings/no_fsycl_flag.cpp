// Test to verify that a warning is thrown when the -fsycl flag is not used and
// <sycl/sycl.hpp> file is included.
// RUN: %clangxx -std=c++17 -I %sycl_include -fsyntax-only %s -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning

// expected-warning@sycl/sycl.hpp:* {{You are including <sycl/sycl.hpp> without -fsycl flag, which is errorenous for device code compilation. This warning can be disabled by setting SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro.}}
#include <sycl/sycl.hpp>
