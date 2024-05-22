// TODO: Passing/returning structures via invoke_simd() API is not implemented
// in GPU driver yet. Enable the test when GPU RT supports it.
// XFAIL: gpu
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/* Test case specification:
 * -----------------------
 * Test passing struct value (uniform) to invoke_simd functions.
 * Test contains 3 different cases with different number of fields in structure.
 * Internally, structures are passed as different basic or std types depending
 * on the number of fields in the structure and whether those fields have
 * matching types.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/usm.hpp>

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

template <class T> struct solo {
  T x;

  float sum() { return x; }
};

template <class TX, class TY> struct duo {
  TX x;
  TY y;

  float sum() { return x + y; }
};

template <class TX, class TY, class TZ> struct trio {
  TX x;
  TY y;
  TZ z;

  float sum() { return x + y + z; }
};

template <class StructTy>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, int i, StructTy S) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a * S.sum();
}

// Specialization for 'Solo' case
template <>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, int i, solo<float> S) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a * S.x;
}

// Specialization for 'Duo' case
template <>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, int i, duo<float, float> S) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a * (S.x + S.y);
}

// Specialization for 'Trio' case
template <>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE(float *A, int i, trio<char, int, float> S) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a * (S.x + S.y + S.z);
}

template <class StructTy>
[[intel::device_indirectly_callable]] simd<float, VL> __regcall SIMD_CALLEE(
    float *A, int i, StructTy S) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE<StructTy>(A, i, S);
  return res;
}

enum StructsTypes { Solo = 1, Duo, Trio, Func };

template <int> class TestID;

using namespace sycl;

template <int CaseNum, StructsTypes UsedStruct> bool test(queue q) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  auto *A = malloc_shared<float>(Size, q);
  auto *C = malloc_shared<float>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
    C[i] = -1;
  }

  // For 'Solo' case
  auto uno = solo<float>{2.0};
  // For 'Duo'
  float X = 1.0;
  // For 'Trio' case
  auto tres = trio<char, int, float>{2, 1, -1.0};
  // For 'Func' case
  int Y = 1;

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<CaseNum>>(
          Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
            sub_group sg = ndi.get_sub_group();
            group<1> g = ndi.get_group();
            uint32_t i = sg.get_group_linear_id() * VL +
                         g.get_group_linear_id() * GroupSize;
            uint32_t wi_id = i + sg.get_local_id();

            float res;

            if constexpr (UsedStruct == StructsTypes::Solo) {
              res = invoke_simd(sg, SIMD_CALLEE<solo<float>>, uniform{A},
                                uniform{i}, uniform{uno});
            } else if constexpr (UsedStruct == StructsTypes::Duo) {
              auto dos = duo<float, float>{X, X};
              res = invoke_simd(sg, SIMD_CALLEE<duo<float, float>>, uniform{A},
                                uniform{i}, uniform{dos});
            } else if constexpr (UsedStruct == StructsTypes::Trio) {
              res = invoke_simd(sg, SIMD_CALLEE<trio<char, int, float>>,
                                uniform{A}, uniform{i}, uniform{tres});
            } else if constexpr (UsedStruct == StructsTypes::Func) {
              auto func = duo<int, int>{Y, Y};
              res = invoke_simd(sg, SIMD_CALLEE<duo<int, int>>, uniform{A},
                                uniform{i}, uniform{func});
            } else {
              static_assert(false, "Unsupported case");
            }

            C[wi_id] = res;
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(A, q);
    sycl::free(C, q);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i)
    if (2.0 * A[i] != C[i])
      err_cnt++;

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    for (unsigned i = 0; i < Size; ++i)
      std::cout << "  data: " << C[i] << ", reference: " << A[i] * (-1) << "\n";
  }

  sycl::free(A, q);
  sycl::free(C, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt == 0;
}

int main(void) {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;
  std::cout << "  Case #1, structure with single element:\n";
  passed &= test<1, StructsTypes::Solo>(q);

  std::cout << "  Case #2, structure with two same type elements:\n";
  passed &= test<2, StructsTypes::Duo>(q);

  std::cout << "  Case #3, structure with tree elements:\n";
  passed &= test<3, StructsTypes::Trio>(q);

  std::cout << "  Case #4, structure with function member being called:\n";
  passed &= test<4, StructsTypes::Func>(q);

  return passed ? 0 : 1;
}
