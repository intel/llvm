// This test checks edge cases handling for std::exp(std::complex<double>) used
// in SYCL kernels.
//
// REQUIRES: aspect-fp64
// UNSUPPORTED: target-amd || target-nvidia
// UNSUPPORTED-INTENDED: This test is intended for backends with SPIR-V support.

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
//
// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

// NOTE: on Windows this test will fail with MSVC 2019 STL headers
// due to a bug in those headers.

#include "exp-std-complex-edge-cases.hpp"

int main() { return test<double>(); }
