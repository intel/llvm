// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * This test checks the case of calling the same external function from the SPMD
 * and ESIMD kernels.
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
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

esimd::simd<float, VL> ESIMD_CALLEE(float *A, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res;
  res.copy_from(A + i);
  return res;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, i);
  return res;
}

using namespace sycl;

int main() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  float *A = static_cast<float *>(malloc_shared(Size * sizeof(float), q));
  float *B = static_cast<float *>(malloc_shared(Size * sizeof(float), q));
  float *C = static_cast<float *>(malloc_shared(Size * sizeof(float), q));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    B[i] = C[i] = -1;
  }

  try {
    sycl::range<1> GlobalRange{Size};
    // Number of workitems in each workgroup.
    sycl::range<1> LocalRange{GroupSize};
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class TestInvokeSimd>(
          Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i = sg.get_group_linear_id() * VL +
                         g.get_group_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();

            float res = invoke_simd(sg, SIMD_CALLEE, uniform{A}, uniform{i});
            B[wi_id] = res;
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

  try {
    sycl::range<1> GlobalRange{Size};
    // Number of workitems in each workgroup.
    sycl::range<1> LocalRange{VL};
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class TestExternalCall>(
          Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            uint32_t i = ndi.get_group(0) * VL;

            esimd::simd<float, VL> res(SIMD_CALLEE(B, i));
            res.copy_to(C + i);
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
    if (A[i] != B[i] || B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << A[i] << " != " << B[i]
                  << " != " << C[i] << "\n";
      }
    }
  }

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt;
}
