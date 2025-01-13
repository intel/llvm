// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

SYCL_EXTERNAL sycl::vec<float, 8> call_fmin_vec(sycl::vec<float, 8> a,
                                                sycl::vec<float, 8> b) {
  return sycl::fmin(a, b);
}

SYCL_EXTERNAL float call_fmin_scalar(float a, float b) {
  return sycl::fmin(a, b);
}
