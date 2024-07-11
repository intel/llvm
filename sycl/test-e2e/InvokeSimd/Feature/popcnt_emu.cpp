// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers

/* Test case purpose:
 * -----------------
 * The purpose of popcnt is to test the population count operation (calculate
 * the number of bits set to 1 in a binary vector) called from withing
 * invoke_simd callee. At the moment most of the functionality that the original
 * test case is supposed to test is not supported by the current implementation
 * of invoke_simd. For example, simd_mask doesn't work in the SIMD or ESIMD
 * functions nor will an element type of bool. This test emulates desired
 * behaviour iterating in a loop over simd of INTs.
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

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

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

// emulating popcount for esimd::simd of ints
int popcnt(esimd::simd<int, VL> mask) {
  int count = 0;
  for (int i = 0; i < VL; i++) {
    if (mask[i]) {
      count++;
    }
  }

  return count;
}

__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE_popcnt(esimd::simd<uint16_t, VL> mask) SYCL_ESIMD_FUNCTION {
  int count = popcnt(mask);
  esimd::simd<int, VL> vc(count);
  return vc;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_popcnt(simd<uint16_t, VL> mask)
        SYCL_ESIMD_FUNCTION;

using namespace sycl;

constexpr bool use_invoke_simd = true;

int main(void) {
  int *output = new int[Size];

  for (unsigned i = 0; i < Size; ++i) {
    output[i] = 0;
  }

  try {
    buffer<int, 1> buf(output, range<1>(Size));

    // We need that many workgroups
    sycl::range<1> GlobalRange{Size};

    // We need that many threads in each group
    sycl::range<1> LocalRange{VL};

    auto q = queue{gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto out_accessor = buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> ndi) SUBGROUP_ATTR {
            sycl::group<1> g = ndi.get_group();
            sycl::sub_group sg = ndi.get_sub_group();
            int id = ndi.get_global_id()[0];
            unsigned int offset = g.get_group_id() * g.get_local_range() +
                                  sg.get_group_id() * sg.get_max_local_range();
            int res;

            if constexpr (use_invoke_simd) {
              // Invoke SIMD function:
              uint16_t dummy = id % 2;
              res = invoke_simd(sg, SIMD_CALLEE_popcnt, dummy);
            } else {
              res = id % 2;
            }
            sg.store(out_accessor.get_multi_ptr<access::decorated::yes>() +
                         offset,
                     res);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    delete[] output;

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (output[i] != VL / 2) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << output[i]
                  << " != " << VL << " / 2"
                  << "\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  delete[] output;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_popcnt(simd<uint16_t, VL> mask)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res_count = ESIMD_CALLEE_popcnt(mask);
  return res_count;
}
