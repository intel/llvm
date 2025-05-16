// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#ifndef __LIBSYCL_TIMESTAMP
#error "__LIBSYCL_TIMESTAMP is expected to be defined by sycl.hpp"
Some weird code to cause compilation error
#endif
