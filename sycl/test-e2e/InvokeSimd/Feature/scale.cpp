// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Tests invoke_simd support in the compiler/headers
 * Test case purpose:
 * -----------------
 * To verify that the simple scale example from the invoke_simd spec
 * https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_invoke_simd.asciidoc
 * works.
 *
 * Test case description:
 * ---------------------
 * Invoke a simple SIMD function that scales all elements of a SIMD type X by a
 * scalar value n.
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

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

constexpr int Size = 512;
constexpr int VL = 16;

/*
 * A simple SIMD function that scales all elements of a SIMD type x by a scalar
 * value n.
 */
template <class T>
__attribute__((always_inline)) esimd::simd<T, VL>
ESIMD_CALLEE_scale(esimd::simd<T, VL> x, T n) SYCL_ESIMD_FUNCTION {
  return x * n;
}

template <class T>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<T, VL> __regcall SIMD_CALLEE_scale(simd<T, VL> x,
                                            T n) SYCL_ESIMD_FUNCTION {
  esimd::simd<T, VL> r = ESIMD_CALLEE_scale<T>(x, n);
  return r;
}

using namespace sycl;

template <class> class TestID;

template <class T> T etalon(T a, T n) { return a * n; }

template <class T, class QueueTY> bool test(QueueTY q) {
  std::cout << "Type: " << typeid(T).name() << "\n";

  T *A = new T[Size];
  T *C = new T[Size];
  for (unsigned i = 0; i < Size; ++i) {
    A[i] = static_cast<T>(i);
    C[i] = static_cast<T>(0);
  }

  // scale factor
  T n = static_cast<T>(2);

  try {
    buffer<T, 1> bufa(A, range<1>(Size));
    buffer<T, 1> bufc(C, range<1>(Size));

    sycl::range<1> GlobalRange{Size};
    sycl::range<1> LocalRange{VL};

    auto e = q.submit([&](handler &cgh) {
      auto acca = bufa.template get_access<access::mode::read>(cgh);
      auto accc = bufc.template get_access<access::mode::write>(cgh);

      cgh.parallel_for<TestID<T>>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> item) SUBGROUP_ATTR {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            unsigned int offset = g.get_group_id() * g.get_local_range() +
                                  sg.get_group_id() * sg.get_max_local_range();

            T va = sg.load(
                acca.template get_multi_ptr<access::decorated::yes>().get() +
                offset);
            T vc = invoke_simd(sg, SIMD_CALLEE_scale<T>, va, uniform{n});
            sg.store(
                accc.template get_multi_ptr<access::decorated::yes>().get() +
                    offset,
                vc);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    delete[] A;
    delete[] C;

    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (C[i] != etalon(A[i], n)) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " * " << n << "\n";
      }
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  delete[] A;
  delete[] C;

  return err_cnt == 0;
}

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  const bool SupportsDouble = dev.has(aspect::fp64);

  bool passed = true;
  passed &= test<unsigned char>(q);
  passed &= test<char>(q);
  passed &= test<unsigned short>(q);
  passed &= test<short>(q);
  passed &= test<unsigned int>(q);
  passed &= test<int>(q);
  passed &= test<unsigned long>(q);
  passed &= test<long>(q);

  passed &= test<float>(q);
  if (SupportsDouble)
    passed &= test<double>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
