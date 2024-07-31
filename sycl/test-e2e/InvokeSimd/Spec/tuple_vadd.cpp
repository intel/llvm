// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers
/* Test case purpose:
 * ------------------
 * To test invoke_simd support for tuples, as defined in the invoke_simd spec.
 *
 * Test case description:
 * ----------------------
 * This test case performs a vector addition of 2 vectors of float, a and b.
 * Instead of passing in 2 separate simd<float, VL> to the SIMD and ESIMD
 * functions, we pass in a single tuple<simd<float, VL>, simd<float, VL>> and
 * then get and add these tuple elements,
 *
 * Implementation notes:
 * --------------------
 * I've included an equivalent set of regular (non-tuple) vadd functions to
 * verify that the overall logic of the program is correct. Switch the
 * invoke_simd() call to see that the regular vadd works correctly.
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/usm.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

#include <tuple>

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

__attribute__((always_inline)) esimd::simd<float, VL> ESIMD_CALLEE_TUPLE_VADD(
    std::tuple<esimd::simd<float, VL>, esimd::simd<float, VL>> tup)
    SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> va = std::get<0>(tup);
  esimd::simd<float, VL> vb = std::get<1>(tup);
  return va + vb;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_TUPLE_VADD(
        std::tuple<simd<float, VL>, simd<float, VL>> tup) SYCL_ESIMD_FUNCTION;

using namespace sycl;

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

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();

        std::tuple<float, float> tup(A[wi_id], B[wi_id]);
        float res = invoke_simd(sg, SIMD_CALLEE_TUPLE_VADD, tup);
        C[wi_id] = res;
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
    simd<float, VL> __regcall SIMD_CALLEE_TUPLE_VADD(
        std::tuple<simd<float, VL>, simd<float, VL>> tup) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE_TUPLE_VADD(tup);
  return res;
}
