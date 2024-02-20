// The tests a basic E2E invoke_simd test checking that
// sycl::address_space_cast works.

// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

#include "../../ESIMD/esimd_test_utils.hpp"

using namespace esimd_test;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

constexpr int VL = 16;

__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, esimd::simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  global_ptr<float, access::decorated::yes> ptr =
      sycl::address_space_cast<access::address_space::global_space,
                               access::decorated::yes, float>(A);
  a.copy_from(ptr + i);
  return a + b;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res;
}

bool test() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q(ESIMDSelector, createExceptionHandler());

  printTestLabel(q);

  float *A = malloc_shared<float>(Size, q);
  float *B = malloc_shared<float>(Size, q);
  float *C = malloc_shared<float>(Size, q);

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
      cgh.parallel_for(
          Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i =
                sg.get_group_linear_id() * VL + g.get_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();
            float res = 0;

            res =
                invoke_simd(sg, SIMD_CALLEE, uniform{A}, B[wi_id], uniform{i});

            C[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    return false;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if ((A[i] + B[i]) != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != 3*("
                  << A[i] << " + " << B[i] << ")\n";
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
  return err_cnt == 0;
}

int main() {
  bool Passed = true;
  Passed &= test();
  return Passed ? 0 : 1;
}
