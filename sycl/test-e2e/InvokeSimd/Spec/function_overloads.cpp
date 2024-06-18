// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers
/*
 * Test case purpose:
 * -----------------
 * To test that function overloads behave as according to the invoke_simd spec
 * https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc.
 *
 * Test case description:
 * ---------------------
 * This test case as been adapted from the test case "scale", which invokes a
 * simple SIMD function that scales all elements of a SIMD type x by a scalar
 * value n. The scale function is overloaded, with one function taking a SIMD
 * and a float, and the other taking 2 SIMDs.
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

constexpr int Size = 1024;
constexpr int VL = 16;

/*
 * An overloaded SIMD function that scales all elements of a SIMD type x by a
 * scalar value n. NOTE: n is passed in as uniform, so x is effecively scaled by
 * n.
 */
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE_scale(esimd::simd<float, VL> x, float n) SYCL_ESIMD_FUNCTION {
  return x * n;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_scale(simd<float, VL> x,
                                                float n) SYCL_ESIMD_FUNCTION;

/*
 * An overloaded SIMD function that effectively does a vector multiplication of
 * vectors x and n.
 */
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE_scale(esimd::simd<float, VL> x,
                   esimd::simd<float, VL> n) SYCL_ESIMD_FUNCTION {
  return x * n;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_scale(
        simd<float, VL> x, simd<float, VL> n) SYCL_ESIMD_FUNCTION;

using namespace sycl;

constexpr bool scale_scalar = true;

int main(void) {
  float *A = new float[Size];
  float *C = new float[Size];

  float n = 2.0;

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    C[i] = 0.0f;
  }

  try {
    buffer<float, 1> bufa(A, range<1>(Size));
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
            float vc;

            // Invoke SIMD function:
            if constexpr (scale_scalar)
              vc = invoke_simd<simd<float, VL> __regcall (*)(
                  simd<float, VL>, float)>(sg, SIMD_CALLEE_scale, va,
                                           uniform{n});
            else
              vc = invoke_simd<simd<float, VL> __regcall (*)(
                  simd<float, VL>, simd<float, VL>)>(sg, SIMD_CALLEE_scale, va,
                                                     n);

            sg.store(PC.get_multi_ptr<access::decorated::yes>().get() + offset,
                     vc);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    delete[] A;
    delete[] C;

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] * n != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " * " << n << "\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  delete[] A;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_scale(simd<float, VL> va,
                                                float n) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE_scale(va, n);
  return res;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_scale(
        simd<float, VL> va, simd<float, VL> n) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE_scale(va, n);
  return res;
}
