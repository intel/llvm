// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira issue resolved
// REQUIRES: TEMPORARY_DISABLED
//
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * Test check basic support of local memory access in invoke_simd.
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

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

class Test {};

using dtype = int;

constexpr int VL = 16;
constexpr int slm_size = 32 * 1024;

ESIMD_INLINE void slm_load_store_test(nd_item<1> *ndi, dtype *A,
                                      dtype *C) SYCL_ESIMD_FUNCTION {
  /* TODO: SLM has to be allocated of outside invoke_simd, but propper
   * intarface is not yet ready. Current test implementation in this regard
   * is a subject to future changes.
   */
  esimd::slm_init<slm_size>();

  esimd::simd<dtype, VL> src_vec(0);
  src_vec.copy_from(A);
  esimd::slm_block_store<dtype, VL>(0, src_vec);

  esimd::simd<dtype, VL> dest_vec(0);
  dest_vec = esimd::slm_block_load<dtype, VL>(0);
  dest_vec.copy_to(C);
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall invoke_slm_load_store_test(
    nd_item<1> *ndi, dtype *A, dtype *C) SYCL_ESIMD_FUNCTION;

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;
  auto *A = malloc_shared<dtype>(VL, q);
  auto *C = malloc_shared<dtype>(VL, q);

  for (auto i = 0; i < VL; i++) {
    A[i] = i;
    C[i] = 0;
  }
  try {
    sycl::nd_range<1> Range({1}, {1});

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<Test>(Range, [=](nd_item<1> item) SUBGROUP_ATTR {
        sycl::group<1> g = item.get_group();
        sycl::sub_group sg = item.get_sub_group();
        invoke_simd(sg, invoke_slm_load_store_test, uniform{&item}, uniform{A},
                    uniform{C});
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(C, q);
    return e.code().value();
  }

  bool pass = true;
  for (auto i = 0; i < VL; i++) {
    if (A[i] != C[i]) {
      std::cout << " C[" << i << "]:" << C[i] << ", A[" << i << "]:" << A[i]
                << std::endl;
      pass = false;
    }
  }

  free(A, q);
  free(C, q);

  std::cout << "Test result: " << (pass ? "Pass" : "Fail") << std::endl;
  return pass ? 0 : 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall invoke_slm_load_store_test(
    nd_item<1> *ndi, dtype *A, dtype *C) SYCL_ESIMD_FUNCTION {
  slm_load_store_test(ndi, A, C);
}
