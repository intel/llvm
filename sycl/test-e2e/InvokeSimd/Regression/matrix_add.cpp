// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

// Tests invoke_simd support in the compiler/headers

/* This program does an integer matrix addition of matrices a + b = c. Each
 * matrix is 256 x 256. Conceptually, this program is very similar to a simple
 * vector addition program, the only difference lies in the specification of the
 * range (now 2D instead of 1D); the actual doVadd() functions remain the same.
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

constexpr int VL = 16;

__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE_doVadd(esimd::simd<int, VL> va,
                    esimd::simd<int, VL> vb) SYCL_ESIMD_FUNCTION {
  return va + vb;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_doVadd(
        simd<int, VL> va, simd<int, VL> vb) SYCL_ESIMD_FUNCTION;

int SPMD_CALLEE_doVadd(int a, int b) { return a + b; }

using namespace sycl;

constexpr bool use_invoke_simd = true;

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();

  // Initialize input and output memory on the host
  constexpr size_t N = 256; // In this 2d (N x N) matrix, there are 256 * 256 =
                            // 65,536 total workitems.
  // NOTE: In this program, the value of B is not really important; the only
  // real reason we are using explicit ND range kernels (i.e., nd_range,
  // nd_item) in this program is because we need to extract the subgroup from
  // nd_item in order to pass it to invoke_simd(). A local range is made and so
  // workitems in the global execution range are grouped into workgroups,
  // however, we do not use any workgroup-specific functionality for the
  // computation.
  // Visualize each workgroup as a B x B sub-matrix.
  constexpr size_t B = 4;
  // The matrix is divided into 4 x 4 = 16 workgroups.
  // workgroup will contain 4 subgroups of 16 workitems each.
  constexpr unsigned GroupSize = B * B;

  // Create and initialize our N x N matrices. Remember that multidimensionality
  // is a programmer convenience implemented on top of an underlying
  // one-dimensional space (pg. 94).
  std::vector<int> a(N * N), b(N * N), c(N * N);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 2);
  std::fill(c.begin(), c.end(), 0);

  {
    // Create buffers associated with inputs and output
    buffer<int, 2> a_buf(a.data(), range<2>(N, N)),
        b_buf(b.data(), range<2>(N, N)), c_buf(c.data(), range<2>(N, N));

    try {
      // Submit the kernel to the queue
      auto e = q.submit([&](handler &h) {
        accessor acc_a{a_buf, h};
        accessor acc_b{b_buf, h};
        accessor acc_c{c_buf, h};

        // START CODE SNIP
        range global{N, N};
        range local{B, B};
        h.parallel_for<class Test>(
            nd_range{global, local}, [=](nd_item<2> ndi) SUBGROUP_ATTR {
              int row_index = ndi.get_global_id(0);
              int column_index = ndi.get_global_id(1);
              int res;

              if constexpr (use_invoke_simd) {
                res = invoke_simd(ndi.get_sub_group(), SIMD_CALLEE_doVadd,
                                  acc_a[row_index][column_index],
                                  acc_b[row_index][column_index]);
              } else {
                res = SPMD_CALLEE_doVadd(acc_a[row_index][column_index],
                                         acc_b[row_index][column_index]);
              }
              acc_c[row_index][column_index] = res;
            }); // END parallel_for()
                // END CODE SNIP
      });       // END submit()
      e.wait();
    } // END try
    catch (sycl::exception const &e) {
      std::cout << "SYCL exception caught: " << e.what() << '\n';
      return e.code().value();
    }
  }

  bool passed = std::all_of(c.begin(), c.end(), [](int i) { return (i == 3); });
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_doVadd(
        simd<int, VL> va, simd<int, VL> vb) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE_doVadd(va, vb);
  return res;
}
