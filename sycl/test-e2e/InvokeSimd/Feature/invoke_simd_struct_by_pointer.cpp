// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers

/* Test case specification:
 * -----------------------
 * Test passing pointers to structs to invoke_simd functions.
 *
 * Test case description:
 * -----------------------------
 * This is a simple test that defines a struct called 'multipliers' with 2 int
 * members, x and y. The pointer to an instance of this structure is passed to
 * a function called by invoke_simd. This function multiplies the vector B by
 * both x and y and adds the resulting value to the vector A. The result of the
 * addition is returned and stored in C.
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

struct multipliers {
  int x;
  int y;
};

/* Performs: A + B * scalars->x * scalars->y. */
__attribute__((always_inline)) esimd::simd<int, VL>
ESIMD_CALLEE(int *A, esimd::simd<int, VL> b, int i,
             multipliers *scalars) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> a;
  a.copy_from(A + i);
  return a + b * scalars->x * scalars->y;
}

[[intel::device_indirectly_callable]] simd<int, VL> __regcall SIMD_CALLEE(
    int *A, simd<int, VL> b, int i, multipliers *scalars) SYCL_ESIMD_FUNCTION;

using namespace sycl;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  auto *A = malloc_shared<int>(Size, q);
  auto *B = malloc_shared<int>(Size, q);
  auto *C = malloc_shared<int>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = -1;
  }

  // USM shared memory allocation for struct multipliers.
  auto *scalars = malloc_shared<multipliers>(Size, q);
  scalars->x = 2;
  scalars->y = 3;

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SUBGROUP_ATTR {
        sub_group sg = ndi.get_sub_group();
        group<1> g = ndi.get_group();
        uint32_t i =
            sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
        uint32_t wi_id = i + sg.get_local_id();
        int res;

        res = invoke_simd(sg, SIMD_CALLEE, uniform{A}, B[wi_id], uniform{i},
                          uniform{scalars});
        C[wi_id] = res;
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    sycl::free(scalars, q);
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i)
    if (A[i] + B[i] * scalars->x * scalars->y != C[i])
      err_cnt++;

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    for (unsigned i = 0; i < Size; ++i)
      std::cout << "  data: " << C[i]
                << ", reference: " << A[i] + B[i] * scalars->x * scalars->y
                << "\n";
  }

  sycl::free(scalars, q);
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}

[[intel::device_indirectly_callable]] simd<int, VL> __regcall SIMD_CALLEE(
    int *A, simd<int, VL> b, int i, multipliers *scalars) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> res = ESIMD_CALLEE(A, b, i, scalars);
  return res;
}
