// This test checks edge cases handling for std::exp(std::complex<double>) used
// in SYCL kernels.
//
// REQUIRES: aspect-fp64
// UNSUPPORTED: hip || cuda
// UNSUPPORTED-INTENDED: This test is intended for backends with SPIR-V support.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "exp-std-complex-edge-cases.hpp"

int main() { return test<double>(); }
