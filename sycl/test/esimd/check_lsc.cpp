// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="error:"
// RUN: not %clangxx %fsycl-host-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="error:"

// This test checks that both host and device compilers can:
// - successfully compile lsc_load2d/lsc_store2d APIs
// - emit an error if some of the restrictions on template parameters are
//   violated

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl;

// --- Postive tests.

template <class T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int NUM_BLOCKS,
          bool TRANSPOSE, bool TRANSFORM, cache_hint L1H, cache_hint L3H,
          int N = __ESIMD_EDNS::get_lsc_block_2d_data_size<
              T, NUM_BLOCKS, BLOCK_HEIGHT, BLOCK_WIDTH, TRANSPOSE>()>
SYCL_EXTERNAL auto test_load(T *ptr, int width, int height,
                             int pitch) SYCL_ESIMD_FUNCTION {
  return lsc_load2d<T, BLOCK_WIDTH, BLOCK_HEIGHT, NUM_BLOCKS, TRANSPOSE,
                    TRANSFORM, L1H, L3H>(ptr, width * sizeof(T) - 1, height - 1,
                                         pitch * sizeof(T) - 1, 0, 0);
}

template <class T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int NUM_BLOCKS,
          cache_hint L1H, cache_hint L3H,
          int N = __ESIMD_EDNS::get_lsc_block_2d_data_size<
              T, NUM_BLOCKS, BLOCK_HEIGHT, BLOCK_WIDTH, false>()>
SYCL_EXTERNAL void test_store(T *ptr, simd<T, N> v, int width, int height,
                              int pitch) SYCL_ESIMD_FUNCTION {
  lsc_store2d<T, BLOCK_WIDTH, BLOCK_HEIGHT, L1H, L3H>(
      ptr, width * sizeof(T) - 1, height - 1, pitch * sizeof(T) - 1, 0, 0, v);
}

// --- Positive tests.

template auto
test_load<float, 16, 16, 1, false, false, cache_hint::none, cache_hint::none>(
    float *, int, int, int) SYCL_ESIMD_FUNCTION;

constexpr int N16_8 =
    __ESIMD_EDNS::get_lsc_block_2d_data_size<float, 1, 16, 8, false>();
template void test_store<float, 8, 16, 1, cache_hint::none, cache_hint::none>(
    float *, simd<float, N16_8>, int, int, int) SYCL_ESIMD_FUNCTION;

// --- Negative tests.

template auto
test_load<float, 32, 32, 1, false, false, cache_hint::none, cache_hint::none>(
    float *, int, int, int) SYCL_ESIMD_FUNCTION;
// CHECK: {{.*}}error: {{.*}}2D load supports 2048 bytes max

constexpr int N16_16 =
    __ESIMD_EDNS::get_lsc_block_2d_data_size<float, 1, 16, 16, false>();
template void test_store<float, 16, 16, 1, cache_hint::none, cache_hint::none>(
    float *, simd<float, N16_16>, int, int, int) SYCL_ESIMD_FUNCTION;
// CHECK: {{.*}}error: {{.*}}Unsupported block width
// CHECK: {{.*}}error: {{.*}}2D store supports 512 bytes max
