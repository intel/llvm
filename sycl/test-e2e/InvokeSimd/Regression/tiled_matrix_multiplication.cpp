// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers

/* This program does an integer matrix multiplication of matrices A * B = C, in
 * a tiled fashion, using work-group local memory as an optimization. "In a
 * tiled fashion" means that each work-item (m, n), which computes the dot
 * product of the row m of A and the column n of B, processes "tiles" (chunks or
 * sub-vectors) of A and B, instead of the whole vector in one go. We introduce
 * invoke_simd when loading each tile of A; each work-item in the work-group
 * loads only its corresponding row element of the current tile of A, and
 * invoke_simd implicitly accumulates, across all the work-items in the
 * work-group, each such row element value into its corresponding position in a
 * SIMD lane, and then the SIMD_CALLEE simply returns this packed SIMD, which
 * then gets implicitly unpacked (serialized), and each row element value gets
 * stored into the tile in local memory. Think of this as the work-items
 * collectively loading the current tile with the help of invoke_simd. Note that
 * this is an artificial and contrived use of invoke_simd, as the alternative,
 * which is just to have each work-item directly load its row element from A and
 * store it directly into its place in the tile, works just as well and is maybe
 * even faster. After loading the tile, each work-item proceeds to independently
 * compute the sub-dot product of the sub-vectors tileA and the sub-vector from
 * the corresponding column of matrix B. Then, this processe is repeated for the
 * next tile, until eventually, all tiles have been processed and their sub-dot
 * products have been accumulated into sum. Then, sum is stored at C[m][n].
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

constexpr int VL = 16;

__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE(esimd::simd<int, VL> va) SYCL_ESIMD_FUNCTION {
  return va;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE(simd<int, VL> va) SYCL_ESIMD_FUNCTION;

int SPMD_CALLEE_doVadd(int a, int b) { return a + b; }

using namespace sycl;

constexpr bool use_invoke_simd = true;
constexpr bool invoke_simd_debug = false; // toogle debug print

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();

  // Initialize input and output memory on the host
  constexpr size_t M = 16 * 2;
  constexpr size_t N = 16 * 2;
  // In this 2d (M x N) matrix, there are 32 * 32 = 1024 total workitems.
  // constexpr size_t N = 256; // In this 2d (N x N) matrix, there are
  // 256 * 256 = 65,536 total workitems.

  // Create and initialize our M x N matrices. Remember that multidimensionality
  // is a programmer convenience implemented on top of an underlying
  // one-dimensional space (pg. 94).
  std::vector<int> a(M * N), b(M * N), c(M * N);
  std::fill(a.begin(), a.end(), 1);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      b[(m * M) + n] = (m * M) + n;
    }
  }
  std::fill(c.begin(), c.end(), 0);

  {
    // Create buffers associated with inputs and output
    buffer<int, 2> a_buf(a.data(), range<2>(M, N)),
        b_buf(b.data(), range<2>(M, N)), c_buf(c.data(), range<2>(M, N));

    try {
      // Submit the kernel to the queue
      auto e = q.submit([&](handler &h) {
        accessor acc_a{a_buf, h};
        accessor acc_b{b_buf, h};
        accessor acc_c{c_buf, h};

        // 1D local accessor, for one matrix tile:
        constexpr int tile_size = 16;
        auto tileA = sycl::local_accessor<int, 1>(tile_size, h);

        // START CODE SNIP
        range global{M, N};
        range local{1, tile_size};
        // range local{B, B};
        h.parallel_for<class Test>(
            nd_range{global, local}, [=](nd_item<2> ndi) SUBGROUP_ATTR {
              // Indices in the global index space:
              int m = ndi.get_global_id(0); // Global row index
              int n = ndi.get_global_id(1); // Global column index

              // Index in the local index space:
              int i = ndi.get_local_id(1); // Local index
              int sum = 0;

              if constexpr (use_invoke_simd) {
                for (int kk = 0; kk < M; kk += tile_size) {
                  // Load the matrix tile from matrix A, and synchronize
                  // to ensure all work-items have a consistent view
                  // of the matrix tile in local memory.
                  // NOTE: I've chosen to introduce the invoke_simd call here.
                  // We implicitly parallelize the loading of the tile; each
                  // work-item i in the work-group loads its corresponding tile
                  // element from matrix A (acc_a) in global memory. This seems
                  // like the only place where it makes sense and is easily
                  // implementable to insert invoke_simd. Remember, invoke_simd
                  // implicitly parallelizes some aspect of the workload across
                  // the work-items belonging to the same workgroup. This seems
                  // like the only aspect that is parallelizable ACROSS the
                  // work-items in a work-group. Ideally, in a matrix multiply,
                  // we want to parallelize the vector dot product computation
                  // carried out by a single work-item, but this is the wrong
                  // level of granularity for invoke_simd; this is
                  // intra-work-item granularity, whereas invoke_simd is
                  // intra-work-group (inter-work-item) granularity.
                  tileA[i] = invoke_simd(ndi.get_sub_group(), SIMD_CALLEE,
                                         acc_a[m][kk + i]);
                  // NOTE: Because invoke_simd implicitly synchronizes the
                  // work-items, we don't have to do so explicitly here.
                  // ndi.barrier();

                  // Perform computation using the local memory tile, and
                  // matrix B in global memory.
                  for (int k = 0; k < tile_size; k++) {
                    sum += tileA[k] * acc_b[kk + k][n];
                  }

                  // After computation, synchronize again, to ensure all
                  // reads from local memory tile are complete.
                  ndi.barrier();
                } // END outer loop
              } else {
                int res = SPMD_CALLEE_doVadd(acc_a[m][n], acc_b[m][n]);
              }
              // Write the final result to global memory.
              acc_c[m][n] = sum;
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

  // Check the result of the matrix multiplication.
  // Simply do the same matrix multiplication and check
  // each result as you go.
  int matrixA[M][N];
  int matrixB[M][N];
  int matrixC[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      matrixA[m][n] = 1;
      matrixB[m][n] = (m * M) + n;
      matrixC[m][n] = 0;
    }
  }

  bool passed = true;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int dotProduct = 0;
      for (int k = 0; k < M; k++) {
        dotProduct += matrixA[m][k] * matrixB[k][n];
      }
      matrixC[m][n] = dotProduct;
      // Check this result against that of the original matrix multiplication
      if (matrixC[m][n] != c[(m * M) + n]) {
        passed = false;
      }

      if constexpr (invoke_simd_debug) {
        if (n == 0) {
          std::cout << '\n';
        }
        std::cout << c[(m * M) + n] << " ";
      }
    }
  }

  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return (passed) ? 0 : 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE(simd<int, VL> va) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE(va);
  return res;
}
