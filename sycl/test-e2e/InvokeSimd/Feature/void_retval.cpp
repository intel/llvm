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
 * To test void return values from a SIMD function. There is no explicit
 * mention of void return values in the invoke_simd spec, but many
 * many workloads (i.e., RenderKit) require such support.
 *
 * Test case description:
 * ---------------------
 * A simple SIMD function that returns a void value.
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
 * A simple SIMD function that ignores it arguments and simply returns void.
 * NOTE: I first tried to wrap void in simd, but the implementation didn't
 * accept that either.
 */
__attribute__((always_inline)) void
ESIMD_CALLEE_return_void(esimd::simd<float, VL> va, esimd::simd<float, VL> vb,
                         float *pvc) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = va + vb;
  esimd::block_store<float, VL>(pvc, res);
  return;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_return_void(
    simd<float, VL> va, simd<float, VL> vb, float *pvc) SYCL_ESIMD_FUNCTION;

using namespace sycl;

constexpr bool use_void = true;

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
            // We need to get a pointer to the starting address of where the
            // result of the vector addition should be stored in/written back to
            // C. Returns the index (ordinal number) of the work-group to which
            // the current work-item belongs (i.e., (work-group) 0, (work-group)
            // 1, etc.). Computes the global work-group starting index; the
            // absolute starting index of the work-group in the ND-range to
            // which the current work-item belongs.
            int group_offset = g.get_group_linear_id() * VL;
            float *pvc =
                PC.get_multi_ptr<access::decorated::yes>().get() + group_offset;

            // Invoke SIMD function:
            // va values from each work-item are combined into a simd<float,
            // VL>.
            invoke_simd(sg, SIMD_CALLEE_return_void, va, vb, uniform{pvc});
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
    if (C[i] != A[i] + B[i]) {
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

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_return_void(
    simd<float, VL> va, simd<float, VL> vb, float *pvc) SYCL_ESIMD_FUNCTION {
  ESIMD_CALLEE_return_void(va, vb, pvc);
  return;
}
