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


#include <sycl/detail/boost/mp11.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

[[intel::device_indirectly_callable]]
simd<float, VL>
SIMD_CALLEE(simd<float, VL> va,
             simd_mask<float, VL> mask) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> ret;
  esimd::simd_mask<VL> emask;
  for(int i = 0; i < VL; i++)
    emask[i] = static_cast<bool>(mask[i]);
  ret.merge(va, !emask);
  return ret;
}

int main() {
  sycl::queue q{gpu_selector_v};
  auto dev = q.get_device();

  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
 constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto ctxt = q.get_context();

  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  bool *M = static_cast<bool *>(malloc_shared(Size * sizeof(bool), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    C[i] = 0;
    M[i] = i % 2;
  }

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(Range, [=](nd_item<1> ndi) {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        float res = invoke_simd(sg, SIMD_CALLEE, A[wi_id], M[wi_id]);
        C[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, q);
    sycl::free(C, q);
    sycl::free(M, q);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if ((i % 2 == 0) && A[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << "\n";
      }
    }
    if((i % 2 == 1) && C[i] > 1/1e8) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != 0\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  sycl::free(A, q);
  sycl::free(C, q);
  sycl::free(M, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt == 0;
}
