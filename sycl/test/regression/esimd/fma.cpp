// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

SYCL_EXTERNAL sycl::vec<float, 8> call_fma_vec(sycl::vec<float, 8> a,
                                               sycl::vec<float, 8> b,
                                               sycl::vec<float, 8> c) {
  return sycl::fma(a, b, c);
}

SYCL_EXTERNAL float call_fma_scalar(float a, float b, float c) {
  return sycl::fma(a, b, c);
}
