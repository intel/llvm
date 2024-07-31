// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
#include <sycl/detail/boost/mp11.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

#include <functional>
#include <iostream>
#include <type_traits>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

[[intel::device_indirectly_callable]] simd<float, VL>
SIMD_CALLEE(simd<float, VL> va, simd_mask<float, VL> mask) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> ret(0);
  esimd::simd_mask<VL> emask = mask;
  ret.merge(va, !emask);
  return ret;
}

int main() {
  sycl::queue q;
  auto dev = q.get_device();

  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  std::array<float, Size> A;
  std::array<float, Size> C;
  std::array<bool, Size> M;

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    C[i] = 0;
    M[i] = i % 2;
  }

  sycl::buffer<float> ABuf(A);
  sycl::buffer<float> CBuf(C);
  sycl::buffer<bool> MBuf(M);

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      sycl::accessor A_acc{ABuf, cgh, sycl::read_only};
      sycl::accessor C_acc{CBuf, cgh, sycl::write_only};
      sycl::accessor M_acc{MBuf, cgh, sycl::read_only};
      cgh.parallel_for(Range, [=](nd_item<1> ndi) {
        sub_group sg = ndi.get_sub_group();
        uint32_t wi_id = ndi.get_global_linear_id();
        float res = invoke_simd(sg, SIMD_CALLEE, A_acc[wi_id], M_acc[wi_id]);
        C_acc[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;
  sycl::host_accessor A_acc(ABuf);
  sycl::host_accessor C_acc(CBuf);

  for (unsigned i = 0; i < Size; ++i) {
    if ((i % 2 == 0) && A_acc[i] != C_acc[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C_acc[i]
                  << " != " << A_acc[i] << "\n";
      }
    }
    if ((i % 2 == 1) && C_acc[i] != 0.0f) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C_acc[i] << " != 0\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  assert(err_cnt == 0);
  return 0;
}
