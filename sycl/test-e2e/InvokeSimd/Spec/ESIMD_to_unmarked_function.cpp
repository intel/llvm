// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers

/* Test case specification:
 * -----------------------
 * Test and report errors if invoked ESIMD function calls SPMD function.
 *
 * Test case description:
 * ---------------------
 * This is an additional test case to increase test coverage. It tests a
 * happy-path. ESIMD_CALLEE_doVadd() calls doVadd(), which is an entirely
 * unmarked function (not tagged with SYCL_EXTERNAL nor SYCL_ESIMD_FUNCTION).
 * Such a function "is treated as ESIMD if called froma within ESIMD code"
 * and is treated as SPMD if called from within SPMD (SYCL) code.
 *
 * Currently, this test case passes because no rules are violated; the unmarked
 * function doVadd() is treated as ESIMD when called from within ESIMD code
 * (ESIMD_CALLEE_doVadd()), and as SPMD when called from within SPMD code.
 * Therefore, we never go from ESIMD to SPMD, or from SPMD to ESIMD in this test
 * case.
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

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

// 1024 / 16 = 64: There will be 64 iterations that process 16 elements each
constexpr int Size = 1024;
// constexpr int Size = 1024 * 128;
constexpr int VL = 16;

/* This unmarked function will be treated the same as its calling context;
 * if it is called from within ESIMD code, it will be treated as ESIMD,
 * if it is called from within SPMD code, it will be treated as SPMD.
 */
// SYCL_EXTERNAL
float doVadd(float a, float b) { return a + b; }

__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE_doVadd(esimd::simd<float, VL> va,
                    esimd::simd<float, VL> vb) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> vc;
  for (int i = 0; i < VL; ++i) {
    float a = va[i];
    float b = vb[i];
    vc[i] = doVadd(a, b);
  }
  return vc;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb) SYCL_ESIMD_FUNCTION;

constexpr bool use_invoke_simd = true;

int main(void) {
  float *A = new float[Size];
  float *B = new float[Size];
  float *C = new float[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  try {
    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));

    // We need that many workgroups
    sycl::range<1> GlobalRange{Size};

    // We need that many threads in each group
    sycl::range<1> LocalRange{VL};

    auto q = queue{gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> item) SUBGROUP_ATTR {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            unsigned int offset = g.get_group_id() * g.get_local_range() +
                                  sg.get_group_id() * sg.get_max_local_range();
            float va = sg.load(
                PA.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vb = sg.load(
                PB.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vc;

            if constexpr (use_invoke_simd) {
              vc = invoke_simd(sg, SIMD_CALLEE_doVadd, va, vb);
            } else {
              vc = doVadd(va, vb);
            }
            sg.store(PC.get_multi_ptr<access::decorated::yes>().get() + offset,
                     vc);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    delete[] A;
    delete[] B;
    delete[] C;

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

  delete[] A;
  delete[] B;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE_doVadd(va, vb);
  return res;
}
