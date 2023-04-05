// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu-intel-pvc
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

// Tests invoke_simd support in the compiler/headers

/* This program tests loop functionality. The key parts of this program
 * are:
 * 1). The global execution range is divided/reduced by VL
 * 2). Each work-item must do VL times more work
 * 3). invoke_simd functionality is integrated into this test case
 *     by accumulating wi_ids into simds instead of vector elements.
 *     There is thus an extra looping step that must occur in order to
 *     perform the vector additions: each simd of wi_ids must be
 *     looped through and each wi_id extracted. Then, each wi_id
 *     is used to compute an absolute offset into the underlying vectors
 *     A, B, and C. That is, each ESIMD function call actually performs
 *     VL vector additions.
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
ESIMD_CALLEE(float *A, float *B, float *C,
             esimd::simd<int, VL> indices) SYCL_ESIMD_FUNCTION {
  // Loop through the indices:
  // Extract each index and use it to load 16 underlying vector elements (from A
  // and B), perform the vector addition of these elements, and store the result
  // in the correct location in C.
  for (int i = 0; i < VL; ++i) {
    int index = indices[i];
    int absolute_index = index * VL;
    esimd::simd<float, VL> a, b;
    a.copy_from(A + absolute_index);
    b.copy_from(B + absolute_index);
    esimd::simd<float, VL> c = a + b;
    c.copy_to(C + absolute_index);
  }

  // The current implementation requires us to return something here; so we
  // return a dummy, which is simply ignored by the caller.
  return esimd::simd<float, VL>();
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, float *B, float *C,
                                          simd<int, VL> indices)
        SYCL_ESIMD_FUNCTION;

void SPMD_doVadd(float va[VL], float vb[VL], float vc[VL]) {
  for (int i = 0; i < VL; i++) {
    vc[i] = va[i] + vb[i];
  }
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

  sycl::range<1> GlobalRange{Size / VL};
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
        int wi_id = i + sg.get_local_id();

        if constexpr (use_invoke_simd) {
          float res = invoke_simd(sg, SIMD_CALLEE, uniform{A}, uniform{B},
                                  uniform{C}, wi_id);
        } else {
          // NOTE: This kernel will be instantiated for each index in the
          // GlobalRange, which is currently 1024 / 16 = 64. However, in
          // this loop implementation, we do not want to do an SPMD-style
          // vector addition on each individual workitem in the global
          // execution range, rather we want to simulate an SIMD-style
          // vector addition using standard arrays of VL workitems, but only
          // processing every VLth index.
          unsigned int offset = ndi.get_global_id(0) * VL;
          float va[VL], vb[VL], vc[VL];
          // Load input vectors A and B.
          for (int k = 0; k < VL; k++) {
            va[k] = A[offset + k];
            vb[k] = B[offset + k];
          }

          SPMD_doVadd(va, vb, vc);
          for (int k = 0; k < VL; k++) {
            C[offset + k] = vc[k];
          }
        }
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
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

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, float *B, float *C,
                                          simd<int, VL> indices)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, B, C, indices);
  return res;
}
