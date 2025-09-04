// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// XFAIL: !system-windows
// UNSUPPORTED: system-windows
// Different versions of STL implementations by Microsoft either implicitly
// include cmath or not. It means the test result depends on the system
// environment.

// TODO: Longer term that header should really be moved to the clang project.
#include <sycl/stl_wrappers/__sycl_cmath_wrapper_impl.hpp>
