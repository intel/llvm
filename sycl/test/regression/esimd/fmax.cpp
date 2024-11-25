// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

SYCL_EXTERNAL sycl::vec<float, 8> call_fmax_vec(sycl::vec<float, 8> a,
                                                sycl::vec<float, 8> b) {
  return sycl::fmax(a, b);
}

SYCL_EXTERNAL float call_fmax_scalar(float a, float b) {
  return sycl::fmax(a, b);
}
