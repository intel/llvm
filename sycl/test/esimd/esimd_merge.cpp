// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel::experimental;
using namespace cl::sycl;

using simd_mask_elem_t = simd_mask<1>::element_type;

template <class T, int N> struct SimdMergeTest {
  inline void test(T *in_a, T *in_b, simd_mask_elem_t *in_mask, T *out)
      __attribute__((sycl_device)) {
    simd<T, N> a(in_a);
    simd<T, N> b(in_b);
    simd_mask<N> m(in_mask);
    simd<T, N> c = esimd::merge(a, b, m);
    c.copy_to(out);
  }
};

template <class T, int N> struct SimdViewMergeTest {
  inline void test(T *in_a, T *in_b, simd_mask_elem_t *in_mask, T *out)
      __attribute__((sycl_device)) {
    simd<T, N> a(in_a);
    simd<T, N> b(in_b);
    simd_mask<N / 2> m(in_mask);
    simd<T, N / 2> c = esimd::merge(a.template select<N / 2, 1>(1),
                                    b.template select<N / 2, 2>(0), m);
    c.copy_to(out);
  }
};

template <class T, int N> struct SimdView2MergeTest {
  inline void test(T *in_a, T *in_b, simd_mask_elem_t *in_mask, T *out)
      __attribute__((sycl_device)) {
    simd<T, N> a(in_a);
    simd<T, N> b(in_b);
    simd_mask<N / 4> m(in_mask);
    simd<T, N / 4> c = esimd::merge(
        a.template select<N / 2, 1>(1).template select<N / 4, 1>(),
        b.template select<N / 2, 2>(0).template select<N / 4, 1>(), m);
    c.copy_to(out);
  }
};

template struct SimdMergeTest<float, 8>;
template struct SimdViewMergeTest<float, 8>;
template struct SimdView2MergeTest<sycl::half, 8>;
