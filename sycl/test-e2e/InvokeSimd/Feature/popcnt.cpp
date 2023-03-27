// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable after simd_mask supported
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

// Tests invoke_simd support in the compiler/headers

/* Test case purpose:
 * -----------------
 * The purpose of popcnt is to test the population count operation (calculate
 * the number of bits set to 1 in a binary vector) called from withing
 * invoke_simd callee.
 *
 * Test case description:
 * ----------------------
 * This program has a single global vector of 1024 ints, and a SIMD width
 * and work-group size of 16. Each work-item sends its (global_id % 2)
 * into an esimd::simd<int, 16>. The ESIMD_CALLEE_popcnt() function
 * counts the number of 1-bits (which result from odd global_ids) in
 * its simd argument. Since the global_ids increase consecutively starting
 * at 0, there should be eight 0-bits and eight 1-bits in any given 16-wide
 * SIMD: [0101010101010101]
 * Therefore, the count of 1-bits will always be the SIMD width / 2, which
 * is then returned to the callee in a SIMD. The final result is the output
 * vector populated with SIMD width / 2 (in this case, 8 since the SIMD width
 * is 16).
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

/* Subgroup size attribute is optional
 * In case it is absent compiler decides what subgroup size to use
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

// 1024 / 16 = 64: There will be 64 iterations that process 16 elements each
constexpr int Size = 1024;
constexpr int VL = 16;

__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE_popcnt(simd_mask<bool, VL> mask) SYCL_ESIMD_FUNCTION {
  // here we call built_in population count operation
  uint32_t count = popcount(mask);
  esimd::simd<int, VL> vc(count);
  return vc;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_popcnt(simd_mask<bool, VL> mask)
        SYCL_ESIMD_FUNCTION;

using namespace sycl;

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctx = q.get_context();

  auto *O =
      static_cast<int32_t *>(malloc_shared(Size * sizeof(int32_t), dev, ctx));
  auto *M = static_cast<bool *>(malloc_shared(Size * sizeof(bool), dev, ctx));

  for (int i = 0; i < Size; ++i) {
    O[i] = -1;
    M[i] = i % 2 == 0;
  }

  try {
    // We need that many workgroups
    sycl::range<1> GlobalRange{Size};

    // We need that many threads in each group
    sycl::range<1> LocalRange{VL};

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> ndi) SUBGROUP_ATTR {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i = sg.get_group_linear_id() * VL +
                         g.get_group_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();

            int res = invoke_simd(sg, SIMD_CALLEE_popcnt, M[wi_id]);
            O[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(O, q);
    sycl::free(M, q);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (O[i] != VL / 2) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << O[i] << " != " << VL
                  << " / 2"
                  << "\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  sycl::free(O, q);
  sycl::free(M, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_popcnt(simd_mask<bool, VL> mask)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE_popcnt(mask);
  return res;
}
