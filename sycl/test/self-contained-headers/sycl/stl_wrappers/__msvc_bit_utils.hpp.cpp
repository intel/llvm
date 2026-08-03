// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// REQUIRES: system-windows
// This wrapper exists only to intercept MSVC's <__msvc_bit_utils.hpp> in SYCL
// device code; on non-Windows there is no such header to wrap.

#include <sycl/stl_wrappers/__msvc_bit_utils.hpp>
