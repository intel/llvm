// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * Test case #1
 * -----------------
 * Purpose:
 * To test uniform (scalar) return values from a SIMD function.
 *
 * The invoke_simd spec:
 * https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc:
 *
 * "Return values of type T are converted to
 * sycl::ext::oneapi::experimental::uniform<T>, and broadcast to each work-item;
 * every work-item in the sub-group receives the same value."
 *
 * Description:
 * A simple SIMD function that returns a uniform (scalar) value as a SIMD type.
 *
 *
 * Test case #2
 * -----------------
 * Purpose:
 * To test uniform (scalar) return values from the invoke_simd spec
 *
 * https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc:
 * "Return values of type T are converted to
 * sycl::ext::oneapi::experimental::uniform<T>, and broadcast to each work-item;
 * every work-item in the sub-group receives the same value."
 *
 * Description:
 * A simple SIMD function that returns a scalar/uniform value as a raw
 * (unwrapped) scalar. In this test case, we simply pass in a uniform type T (n
 * = 2) to the SIMD function and then return it and store it to C[i]. Therefore,
 * C[i] == n for all i.
 *
 * This test also runs with all types of VISA link time optimizations enabled.
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

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

// 1024 / 16 = 64: There will be 128 iterations that process 8 elements each
constexpr int Size = 1024;
constexpr int VL = 16;

/*
 * A simple SIMD function that takes a SIMD type x and a scalar value n, and
 * simply returns the scalar, which should be broadcast to each work-item; every
 * work-item in the sub-group receives the same value. NOTE: I'm not sure if the
 * return value type is supposed to be declared as uniform{} or type T, but
 * returning the scalar as a SIMD type seems to work fine.
 */
template <class T>
__attribute__((always_inline)) esimd::simd<T, VL>
ESIMD_CALLEE_return_uniform_SIMD(esimd::simd<T, VL> x,
                                 T n) SYCL_ESIMD_FUNCTION {
  return n;
}

template <class T>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<T, VL> __regcall SIMD_CALLEE_return_uniform_SIMD(simd<T, VL> x, T n)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<T, VL> r = ESIMD_CALLEE_return_uniform_SIMD<T>(x, n);
  return r;
}

/*
 * A simple SIMD function that takes a SIMD type x and a scalar value n, and
 * simply returns the scalar, which should be broadcast to each work-item; every
 * work-item in the sub-group receives the same value. NOTE: I'm not sure if the
 * return value type is supposed to be declared as uniform{} or type T, but
 * returning the scalar as a SIMD type seems to work fine.
 */
template <class T>
__attribute__((always_inline)) T
ESIMD_CALLEE_return_uniform_scalar(esimd::simd<T, VL> x,
                                   T n) SYCL_ESIMD_FUNCTION {
  return n;
}

template <class T>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    T __regcall SIMD_CALLEE_return_uniform_scalar(simd<T, VL> x,
                                                  T n) SYCL_ESIMD_FUNCTION {
  T r = ESIMD_CALLEE_return_uniform_scalar<T>(x, n);
  return r;
}

using namespace sycl;

template <class, bool> class TestID;

template <class T, bool return_SIMD, class QueueTY> bool test(QueueTY q) {
  std::cout << "Type: " << typeid(T).name() << ", returning "
            << (return_SIMD ? "SIMD" : "scalar") << "\n";

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

      cgh.parallel_for<TestID<T, return_SIMD>>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> item) SUBGROUP_ATTR {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            unsigned int offset = g.get_group_id() * g.get_local_range() +
                                  sg.get_group_id() * sg.get_max_local_range();

            T va = sg.load(acca.get_pointer() + offset);
            T vc;

            if constexpr (return_SIMD)
              vc = invoke_simd(sg, SIMD_CALLEE_return_uniform_SIMD<T>, va,
                               uniform{n});
            else
              vc = invoke_simd(sg, SIMD_CALLEE_return_uniform_scalar<T>, va,
                               uniform{n});

            sg.store(accc.get_pointer() + offset, vc);
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
    if (C[i] != n) {
      if (++err_cnt < 10) {
        /*
          std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                    << " * " << n << "\n";
        */
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

  bool passed = true;

  // return SIMD
  passed &= test<unsigned char, true>(q);
  passed &= test<char, true>(q);
  passed &= test<unsigned short, true>(q);
  passed &= test<short, true>(q);
  passed &= test<unsigned int, true>(q);
  passed &= test<int, true>(q);
  passed &= test<unsigned long, true>(q);
  passed &= test<long, true>(q);

  passed &= test<float, true>(q);
  // double type not supported by most platforms
  // passed &= test<double, true>(q);

  // return scalar
  passed &= test<unsigned char, false>(q);
  passed &= test<char, false>(q);
  passed &= test<unsigned short, false>(q);
  passed &= test<short, false>(q);
  passed &= test<unsigned int, false>(q);
  passed &= test<int, false>(q);
  passed &= test<unsigned long, false>(q);
  passed &= test<long, false>(q);

  passed &= test<float, false>(q);
  // double type not supported by most platforms
  // passed &= test<double, false>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
