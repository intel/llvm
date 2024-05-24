// REQUIRES: gpu-intel-dg1 || gpu-intel-dg2 || gpu-intel-pvc
//
// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/* The test checks that invoke_simd implementation performs proper call of
 * ESIMD math function dp4a.
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

/* Subgroup size attribute is optional
 * In case it is absent compiler decides what subgroup size to use
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(SIZE)]]
#endif

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

constexpr unsigned SIZE = 16;
constexpr unsigned GROUPSIZE = 1;
using DTYPE = unsigned int;

ESIMD_INLINE
esimd::simd<DTYPE, SIZE>
ESIMD_CALLEE(esimd::simd<DTYPE, SIZE> src0, esimd::simd<DTYPE, SIZE> src1,
             esimd::simd<DTYPE, SIZE> src2) SYCL_ESIMD_FUNCTION {
  auto res = esimd::dp4a<DTYPE, DTYPE, DTYPE, DTYPE, SIZE>(src0, src1, src2);
  return res;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<DTYPE, SIZE> __regcall SIMD_CALLEE(
        simd<DTYPE, SIZE> src0, simd<DTYPE, SIZE> src1,
        simd<DTYPE, SIZE> src2) SYCL_ESIMD_FUNCTION;

using namespace sycl;

int main() {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctx = q.get_context();

  DTYPE *S0 =
      static_cast<DTYPE *>(malloc_shared(SIZE * sizeof(DTYPE), dev, ctx));
  DTYPE *S1 =
      static_cast<DTYPE *>(malloc_shared(SIZE * sizeof(DTYPE), dev, ctx));
  DTYPE *S2 =
      static_cast<DTYPE *>(malloc_shared(SIZE * sizeof(DTYPE), dev, ctx));

  DTYPE *RES =
      static_cast<DTYPE *>(malloc_shared(SIZE * sizeof(DTYPE), dev, ctx));

  for (unsigned i = 0; i < SIZE; ++i) {
    S0[i] = 0x32;
    S1[i] = 0x0102037F;
    S2[i] = 0x0102037F;
    RES[i] = 0xcafe;
  }

  sycl::range<1> GroupRange{SIZE};
  sycl::range<1> TaskRange{GROUPSIZE};
  sycl::nd_range<1> Range(GroupRange, TaskRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        uint32_t wi_id = ndi.get_global_linear_id();

        DTYPE res =
            invoke_simd(sg, SIMD_CALLEE, S0[wi_id], S1[wi_id], S2[wi_id]);
        RES[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(S0, ctx);
    sycl::free(S1, ctx);
    sycl::free(S2, ctx);
    sycl::free(RES, ctx);
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < SIZE; ++i) {
    if (RES[i] != 0x3F41) {
      err_cnt++;
      std::cout << "failed at index " << i << ", " << RES[i] << " != 0x"
                << std::hex << 0x3F41 << std::dec << "\n";
    }
  }

  sycl::free(S0, ctx);
  sycl::free(S1, ctx);
  sycl::free(S2, ctx);
  sycl::free(RES, ctx);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<DTYPE, SIZE> __regcall SIMD_CALLEE(
        simd<DTYPE, SIZE> src0, simd<DTYPE, SIZE> src1,
        simd<DTYPE, SIZE> src2) SYCL_ESIMD_FUNCTION {
  esimd::simd<DTYPE, SIZE> res = ESIMD_CALLEE(src0, src1, src2);
  return res;
}
