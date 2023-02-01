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

/* This program does an integer matrix multiplication of matrices a x b = c.
 * Each matrix is 256 x 256. The purpose of this program is to test invoke_simd
 * support. invoke_simd is a mechanism to invoke explicitly vectorized functions
 * written using SIMD types from a SYCL kernel (SPMD context).
 *
 * There are potentially several different ways to craft a matrix multiplication
 * to fit invoke_simd. The approach used in this program is to pack a pair of
 * simd lanes with multiple row indices and column indices respectively, i.e.,
 * simd row_indices = [row0, row1, row2, ..., rowN] and simd column_indices =
 * [col0, col1, col2, ..., colN]. We use these simd lanes as parallel arrays and
 * compute a dot product for each (row, col) pair. This means that each
 * invocation of the ESIMD_CALLEE is responsible for computing VL separate dot
 * products, each of which is packed back into a simd lane which is then
 * returned to the caller in the SPMD context. Each dot product is returned
 * individually to the caller in the SPMD context.
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
constexpr size_t N = 16;

using namespace sycl; // For accessor.

/* This invoke_simd-style function is called by the kernel to perform VL
 * separate matrix multiplications (compute VL separate dot products) on behalf
 * of each work-item (whose indices are passed in as arguments that act as
 * parallel arrays whereby each pair of row_indices[i] and column_indices[i]
 * represents an individual work-item from the execution range). For each such
 * pair (work-item), we compute the corresponding dot product using the
 * accessors acc_a and acc_b which wrap the underlying matrices a and b. Each
 * resulting dot product is stored in its position in the the simd vector, which
 * is returned from this function.
 */
__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE_computeDotProducts(esimd::simd<int, VL> row_indices,
                                esimd::simd<int, VL> column_indices, int *acc_a,
                                int *acc_b) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> vc;
  // Effectively unpack each simd argument to extract each individual
  // (row_index, column_index) pair. For each pair, compute its dot product from
  // the input matrices a and b.
  for (int i = 0; i < VL; ++i) {
    int dot_product = 0;
    int row_index = row_indices[i];
    int column_index = column_indices[i];
    // Now that we have a (row_index, column_index) pair, compute its dot
    // product from the input matrices a and b.
    for (int k = 0; k < N; ++k) {
      dot_product += acc_a[(row_index * N) + k] * acc_b[(k * N) + column_index];
    }
    // Store the dot_product of the current (row_index, column_index) pair in
    // the corresponding position of the output simd (vector).
    vc[i] = dot_product;
  }

  return vc;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE_computeDotProducts(
        simd<int, VL> row_indices, simd<int, VL> column_indices, int *acc_a,
        int *acc_b) SYCL_ESIMD_FUNCTION;

int SPMD_doVmultiply(int a, int b) { return a * b; }

constexpr bool use_invoke_simd = true;

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();

  // Initialize input and output memory on the host
  // NOTE: In this program, the value of B is not really important (i.e., it is
  // more or less an arbitrary value); the only reason we are using explicit ND
  // range kernels (i.e., nd_range, nd_item) in this program is because we need
  // to extract the subgroup from the nd_item in order to pass it to
  // invoke_simd(), which requires a subgroup argument. A local (execution)
  // range is defined and consequently workitems in the global execution range
  // are grouped into workgroups, however, we do not use any this grouping or
  // any workgroup-specific functionality for this computation (i.e., workgroup
  // local memory, work-item communication or synchronization within a
  // workgroup, etc.).
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

              if constexpr (use_invoke_simd) {
                int res = invoke_simd(
                    ndi.get_sub_group(), SIMD_CALLEE_computeDotProducts,
                    row_index, column_index, uniform{acc_a.get_pointer()},
                    uniform{acc_b.get_pointer()});
                acc_c[row_index][column_index] = res;
              } else {
                for (int k = 0; k < N; ++k) {
                  acc_c[row_index][column_index] += SPMD_doVmultiply(
                      acc_a[row_index][k], acc_b[k][column_index]);
                }
              }
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

  // Check that all outputs match serial execution.
  bool passed = true;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      float gold = 0;
      for (int k = 0; k < N; ++k) {
        gold += a[j * N + k] * b[k * N + i];
      }
      if (std::abs(gold - c[j * N + i]) / gold > 1.0E-06) {
        passed = false;
      }
    }
  }

  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;

  // DEBUG: Print failing results
  if (!passed) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << c[(i * N) + j] << ((j == N - 1) ? '\n' : ' ');
      }
    }
  }

  return (passed) ? 0 : 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL simd<int, VL>
SIMD_CALLEE_computeDotProducts(simd<int, VL> row_indices,
                               simd<int, VL> column_indices, int *acc_a,
                               int *acc_b) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE_computeDotProducts(
      row_indices, column_indices, acc_a, acc_b);
  return res;
}
