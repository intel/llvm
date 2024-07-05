// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers
/* Test case description:
 * ----------------------
 * This is a minimal test case to test invoke_simd support for simd_mask,
 * as defined in the invoke_simd spec.
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include <sycl/detail/boost/mp11.hpp>
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
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

template <typename MaskType>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(esimd::simd<float, VL> va,
             simd_mask<MaskType, VL> mask) SYCL_ESIMD_FUNCTION {
  return va;
}

template <typename MaskType>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(
        simd<float, VL> va, simd_mask<MaskType, VL> mask) SYCL_ESIMD_FUNCTION;

using namespace sycl;

template <typename MaskType> int test(queue q) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto dev = q.get_device();
  auto ctxt = q.get_context();

  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  bool *M = static_cast<bool *>(malloc_shared(Size * sizeof(bool), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    C[i] = -1;

    M[i] = true;
  }

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        auto Callee = SIMD_CALLEE<MaskType>;
        float res = invoke_simd(sg, Callee, A[wi_id], M[wi_id]);
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
    if (A[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << "\n";
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
  return err_cnt > 0 ? 1 : 0;
}

template <typename MaskType>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(
        simd<float, VL> va, simd_mask<MaskType, VL> mask) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(va, mask);
  return res;
}

int main() {
  queue q{gpu_selector_v};

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;
  const bool SupportsDouble = dev.has(aspect::fp64);
  using namespace sycl::detail::boost::mp11;
  using MaskTypes =
      std::tuple<char, char16_t, char32_t, wchar_t, signed char, signed short,
                 signed int, signed long, signed long long, unsigned char,
                 unsigned short, unsigned int, unsigned long,
                 unsigned long long, float, double>;
  tuple_for_each(MaskTypes{}, [&](auto &&x) {
    using T = std::remove_reference_t<decltype(x)>;
    if (std::is_same_v<T, double> && !SupportsDouble)
      return;
    passed &= !test<T>(q);
  });
  std::cout << (passed ? "Test passed\n" : "TEST FAILED\n");
  return passed ? 0 : 1;
}
