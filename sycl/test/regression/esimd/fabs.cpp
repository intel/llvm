// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

SYCL_EXTERNAL sycl::vec<float, 8> call_fabs_vec(sycl::vec<float, 8> input) {
  return sycl::fabs(input);
}

SYCL_EXTERNAL float call_fabs_scalar(float input) { return sycl::fabs(input); }
