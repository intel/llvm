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

/* Test case specification:
 * -----------------------
 * Test passing pointers to structs to invoke_simd functions.
 *
 * Test case description:
 * -----------------------------
 * This is a simple test case that defines a struct named
 * multipliers with 2 int members, x and y. Suppose these
 * members represent values by which we want to scale the
 * addtion of vectors A + B. That is, we want to compute
 * C = A + B * x * y (the sum of A + B scaled by x,
 * and then scaled by y).
 *
 * Test case implementation notes:
 * -------------------------------
 * This test case is a modified
 * https://github.com/intel/llvm/blob/sycl/sycl/test/invoke_simd/invoke_simd.cpp
 * It simply extends it by adding a struct pointer parameter to the invoke_simd
 * functions which use it to scale the original vector addition.
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

struct multipliers {
  int x;
  int y;
};

/* Performs the addition of vectors A + B scaled by scalars->x and scalars->y.*/
__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE(int *A, esimd::simd<int, VL> b, int i,
             multipliers *scalars) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> a;
  a.copy_from(A + i);
  // Add vectors A + B and scale them by x and y.
  return a + b * scalars->x * scalars->y;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE(int *A, simd<int, VL> b, int i,
                                        multipliers *scalars)
        SYCL_ESIMD_FUNCTION;

int SPMD_CALLEE(int *A, int b, int i, multipliers *scalars) {
  return A[i] + b * scalars->x * scalars->y;
}

using namespace sycl;

constexpr bool use_invoke_simd = true;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctx = q.get_context();

  int *A = static_cast<int *>(malloc_shared(Size * sizeof(int), dev, ctx));
  int *B = static_cast<int *>(malloc_shared(Size * sizeof(int), dev, ctx));
  int *C = static_cast<int *>(malloc_shared(Size * sizeof(int), dev, ctx));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  // USM shared memory allocation for a struct multipliers.
  multipliers *scalars = static_cast<multipliers *>(
      malloc_shared(/*Size **/ sizeof(multipliers), dev, ctx));
  scalars->x = 2;
  scalars->y = 3;

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        int res;

        if constexpr (use_invoke_simd) {
          res = invoke_simd(sg, SIMD_CALLEE, uniform{A}, B[wi_id], uniform{i},
                            uniform{scalars});
        } else {
          res = SPMD_CALLEE(A, B[wi_id], wi_id, scalars);
        }
        C[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(scalars, ctx);
    sycl::free(A, ctx);
    sycl::free(B, ctx);
    sycl::free(C, ctx);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] * scalars->x * scalars->y != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  sycl::free(scalars, ctx);
  sycl::free(A, ctx);
  sycl::free(B, ctx);
  sycl::free(C, ctx);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<int, VL> __regcall SIMD_CALLEE(int *A, simd<int, VL> b, int i,
                                        multipliers *scalars)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE(A, b, i, scalars);
  return res;
}
