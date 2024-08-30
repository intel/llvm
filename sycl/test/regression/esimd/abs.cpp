// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

SYCL_EXTERNAL sycl::vec<int, 8> call_abs_vec(sycl::vec<int, 8> input) {
  return sycl::abs(input);
}

SYCL_EXTERNAL int call_abs_scalar(int input) { return sycl::abs(input); }
