// RUN: %clangxx -fsycl -fsyntax-only %s
// expected-no-diagnostics

#include <sycl/ext/intel/experimental/esimd.hpp>

// This test checks that host compiler can compile esimd.hpp when there is no
// sycl.hpp included (can happen in pure-ESIMD library).
