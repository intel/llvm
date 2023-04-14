// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * This test checks the case of calling an external function from the SPMD and
 * the one nested in the first one.
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

/* Subgroup size attribute is optional.
 * In case it is absent compiler decides what subgroup size to use.
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

/* *** */

esimd::simd<float, VL>
ESIMD_CALLEE_DEC(esimd::simd<float, VL> a) SYCL_ESIMD_FUNCTION {
  return a - 1;
}

esimd::simd<float, VL>
ESIMD_CALLEE_INC(esimd::simd<float, VL> a) SYCL_ESIMD_FUNCTION {
  return a + 1;
}

[[intel::device_indirectly_callable]] simd<float, VL> __regcall SIMD_CALLEE_INC(
    float *A, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return ESIMD_CALLEE_INC(a);
}

/* *** */

esimd::simd<float, VL> ESIMD_CALLEE_ATTUNITE(esimd::simd<float, VL> a,
                                             bool flag) SYCL_ESIMD_FUNCTION {
  return flag ? ESIMD_CALLEE_INC(a) : ESIMD_CALLEE_DEC(a);
}

[[intel::device_indirectly_callable]] simd<float, VL> __regcall SIMD_CALLEE(
    float *A, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a = SIMD_CALLEE_INC(A, i);
  return ESIMD_CALLEE_ATTUNITE(a, i < 0);
}

/* *** */

using namespace sycl;

int main() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  auto *A = malloc_shared<float>(Size, q);
  auto *B = malloc_shared<float>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    B[i] = -1;
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
            res += invoke_simd(sg, SIMD_CALLEE_INC, uniform{A}, uniform{i});
            B[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, q);
    sycl::free(B, q);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i)
    if (1 + 2 * A[i] != B[i])
      err_cnt++;

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    for (unsigned i = 0; i < Size; ++i)
      std::cout << "  data: " << B[i] << ", reference: " << 1 + 2 * A[i]
                << "\n";
  }

  sycl::free(A, q);
  sycl::free(B, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt;
}
