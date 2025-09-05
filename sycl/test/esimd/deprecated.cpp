// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

void test_slm_init() SYCL_ESIMD_FUNCTION { slm_init(1024); }

void test_fence() SYCL_ESIMD_FUNCTION {
  fence<fence_mask::global_coherent_fence | fence_mask::local_barrier>();
}

void test_split_barrier() SYCL_ESIMD_FUNCTION {
  split_barrier<split_barrier_action::signal>();
}
