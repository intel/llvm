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

/* Test multiple SPDM callers invoke the same esimd functions at different call
 * sites. Here is one possible interpretation of the above: This program has 2
 * kernels, which execute one after the other. The first kernel does the
 * following vector addition: C = A + A. The second kernel does the following
 * vector addition: C = A + C, or C = A + (A + A), or C = A * 3. Currently, the
 * SPMD version of this program works correctly, but the invoke_simd version
 * does not.
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

__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(esimd::simd<float, VL> a,
             esimd::simd<float, VL> b) SYCL_ESIMD_FUNCTION {
  return a + b;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(simd<float, VL> a, simd<float, VL> b)
        SYCL_ESIMD_FUNCTION;

float SPMD_CALLEE(float va, float vb) { return va + vb; }

using namespace sycl;

constexpr bool use_invoke_simd = true;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();

  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  /* Kernel 1: This kernel does the following vector addition: C = A + A. */
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test1>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        float res;

        if constexpr (use_invoke_simd) {
          res = invoke_simd(sg, SIMD_CALLEE, A[wi_id], A[wi_id]);
        } else {
          res = SPMD_CALLEE(A[wi_id], A[wi_id]);
        }
        C[wi_id] = res;
      }); // END parallel_for()
    });   // END queue.submit()
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, ctxt);
    sycl::free(B, ctxt);
    sycl::free(C, ctxt);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  /* Kernel 2: In this second kernel, we do another vector add of A and C
   * (remember that C contains the sum of vector addition from the first kernel:
   * A + A). Effectively, this program computes C = A + A + A, or C = A * 3. */
  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test2>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        float res;

        if constexpr (use_invoke_simd) {
          res = invoke_simd(sg, SIMD_CALLEE, A[wi_id], C[wi_id]);
        } else {
          res = SPMD_CALLEE(A[wi_id], C[wi_id]);
        }
        C[wi_id] = res;
      }); // END parallel_for()
    });   // END queue.submit()
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, ctxt);
    sycl::free(B, ctxt);
    sycl::free(C, ctxt);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (C[i] != A[i] * 3) {
      // if (A[i] + B[i] != C[i]) {
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

  sycl::free(A, ctxt);
  sycl::free(B, ctxt);
  sycl::free(C, ctxt);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(simd<float, VL> a, simd<float, VL> b)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(a, b);
  return res;
}
