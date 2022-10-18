// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

// dummy use of 'v' - storing it to memory
#define USE_v *(decltype(v) *)(++out) = v

// --- bitwise
template <class T1, class T2>
[[intel::sycl_explicit_simd]] auto
bitwise_op_test_impl(const simd<T1, 8> &x, simd<T1, 8> &x1,
                     const simd<T2, 8> &y, simd<T2, 8> &y1,
                     const simd_mask<8> &m, simd_mask<8> &m1,
                     simd<long long, 32> *out) {
  int a = 1;

  // simd ^ simd
  {
    auto k = x1 ^= y;
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x ^ y;
    USE_v;
  }
  // simd ^ SCALAR
  {
    auto k = x1 ^= 5;
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x ^ (T2)5;
    USE_v;
  }
  // SCALAR ^ simd
  {
    auto v = (T1)5 ^ y;
    USE_v;
  }

  // mask ^ mask
  {
    auto k = m1 ^= m;
    auto v = k ^ m;
    USE_v;
  }
  {
    auto v = m ^ m1;
    USE_v;
  }
  // mask ^ SCALAR
  {
    auto k = m1 ^= a;
    auto v = m ^ k;
    USE_v;
  }
  {
    auto v = m ^ 5;
    USE_v;
  }
  // SCALAR ^ mask
  {
    auto v = 5 ^ m;
    USE_v;
  }

  // simd_view<simd,...> ^ simd_view<simd,...>
  {
    simd<T1, 8> k = x1.template select<8, 1>() ^= y1.template select<8, 1>();
    auto v = k ^ y;
    USE_v;
  }
  {
    simd<T1, 8> k = x1.template select<8, 1>().template select<8, 1>() ^=
        y1.template select<8, 1>().template select<8, 1>();
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() ^ y1.template select<8, 1>();
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>().template select<8, 1>() ^
             y1.template select<8, 1>().template select<8, 1>();
    USE_v;
  }
  // simd ^ simd_view<simd,...>
  {
    auto k = x1 ^= y1.template select<8, 1>();
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x ^ y1.template select<8, 1>();
    USE_v;
  }
  // simd_view<simd,...> ^ simd
  {
    simd<T1, 8> k = x1.template select<8, 1>() ^= y;
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() ^ y;
    USE_v;
  }

  // simd_view<simd_mask,...> ^ simd_view<simd_mask,...>
  {
    simd_mask<8> k = m1.select<8, 1>() ^= m1.select<8, 1>();
    auto v = k ^ m;
    USE_v;
  }
  {
    simd_mask<8> k = m1.select<8, 1>().select<8, 1>() ^=
        m1.select<8, 1>().select<8, 1>();
    auto v = k ^ m;
    USE_v;
  }
  {
    auto v = m1.select<8, 1>() ^ m1.select<8, 1>();
    USE_v;
  }
  {
    auto v =
        m1.select<8, 1>().select<8, 1>() ^ m1.select<8, 1>().select<8, 1>();
    USE_v;
  }
  // simd_mask ^ simd_view<simd_mask,...>
  {
    auto k = m1 ^= m1.select<8, 1>();
    auto v = k ^ m;
    USE_v;
  }
  {
    auto v = m ^ m1.select<8, 1>();
    USE_v;
  }
  {
    auto v = m ^ m1.select<8, 1>().select<8, 1>();
    USE_v;
  }
  // simd_view<simd_mask,...> ^ simd_mask
  {
    simd_mask<8> k = m1.select<8, 1>() ^= m;
    auto v = k ^ m;
    USE_v;
  }
  {
    auto v = m1.select<8, 1>() ^ m;
    USE_v;
  }
  {
    auto v = m1.select<8, 1>().select<8, 1>() ^ m;
    USE_v;
  }

  // simd_view<simd,...> ^ SCALAR
  {
    simd<T1, 8> k = x1.template select<8, 1>() ^= (T2)5;
    auto v = k ^ y;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() ^ (T2)5;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>().template select<8, 1>() ^ (T2)5;
    USE_v;
  }
  // SCALAR ^ simd_view<simd,...>
  {
    auto v = (T1)5 ^ y1.template select<8, 1>();
    USE_v;
  }
  {
    auto v = (T1)5 ^ y1.template select<8, 1>().template select<8, 1>();
    USE_v;
  }

  // simd_view<simd_mask,...> ^ SCALAR
  {
    simd_mask<8> k = m1.template select<8, 1>() ^= a;
    auto v = k ^ m;
    USE_v;
  }
  {
    auto v = m1.template select<8, 1>() ^ a;
    USE_v;
  }
  {
    auto v = m1.template select<8, 1>().template select<8, 1>() ^ a;
    USE_v;
  }
  // SCALAR ^ simd_view<simd_mask,...>
  {
    auto v = a ^ y1.template select<8, 1>();
    USE_v;
  }
  {
    auto v = a ^ y1.template select<8, 1>().template select<8, 1>();
    USE_v;
  }
}

template <class T1, class T2>
[[intel::sycl_explicit_simd]] void bitwise_op_test(simd<long long, 32> *out) {
  simd<T1, 8> x((T1)10);
  simd<T1, 8> x1((T1)11);
  const simd<T2, 8> y((T2)17);
  simd<T2, 8> y1((T2)19);
  const simd_mask<8> m(1);
  simd_mask<8> m1(0);

  bitwise_op_test_impl(x, x1, y, y1, m, m1, out);
}

[[intel::sycl_explicit_simd]] void bitwise_op_tests(simd<long long, 32> *out) {
  bitwise_op_test<unsigned char, long long>(out);
  bitwise_op_test<unsigned char, char>(out);
  bitwise_op_test<int, unsigned short>(out);
}

// --- arithmetic
template <class T1, class T2>
[[intel::sycl_explicit_simd]] auto
arith_bin_op_test_impl(const simd<T1, 8> &x, simd<T1, 8> &x1,
                       const simd<T2, 8> &y, simd<T2, 8> &y1,
                       simd<long long, 32> *out) {
  // simd * simd
  {
    auto k = x1 *= y;
    auto v = x * k;
    USE_v;
  }
  {
    auto v = x * y;
    USE_v;
  }
  // simd * SCALAR
  {
    auto k = x1 *= (T2)5;
    auto v = x * k;
    USE_v;
  }
  {
    auto v = x * (T2)5;
    USE_v;
  }
  // SCALAR * simd
  {
    auto v = (T1)5 * y;
    USE_v;
  }

  // simd_view<simd,...> * simd_view<simd,...>
  {
    simd<T1, 8> k = x1.template select<8, 1>() *= y1.template select<8, 1>();
    auto v = x1.template select<8, 1>() * k;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() * y1.template select<8, 1>();
    USE_v;
  }
  // simd * simd_view<simd,...>
  {
    auto k = x1 *= y1.template select<8, 1>();
    auto v = x * k;
    USE_v;
  }
  {
    auto v = x * y1.template select<8, 1>();
    USE_v;
  }
  // simd_view<simd,...> * simd
  {
    simd<T1, 8> k = x1.template select<8, 1>() *= y;
    auto v = k * y;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() * y;
    USE_v;
  }

  // simd_view<simd,...> * SCALAR
  {
    simd<T1, 8> k = x1.template select<8, 1>() *= (T2)5;
    auto v = k * (T2)5;
    USE_v;
  }
  {
    auto v = x1.template select<8, 1>() * (T2)5;
    USE_v;
  }
  // SCALAR * simd_view<simd,...>
  {
    auto v = (T1)5 * y1.template select<8, 1>();
    USE_v;
  }
}

template <class T1, class T2>
[[intel::sycl_explicit_simd]] void arith_bin_op_test(simd<long long, 32> *out) {
  simd<T1, 8> x((T1)10);
  simd<T1, 8> x1((T1)11);
  const simd<T2, 8> y((T2)17);
  simd<T2, 8> y1((T2)19);

  arith_bin_op_test_impl(x, x1, y, y1, out);
}

[[intel::sycl_explicit_simd]] void
arith_bin_op_tests(simd<long long, 32> *out) {
  arith_bin_op_test<unsigned char, long long>(out);
  arith_bin_op_test<unsigned char, char>(out);
  arith_bin_op_test<int, unsigned short>(out);
  arith_bin_op_test<float, char>(out);
  arith_bin_op_test<float, float>(out);
}

// --- equality comparison

template <class T1, class T2>
[[intel::sycl_explicit_simd]] auto
equ_cmp_test_impl(const simd<T1, 8> &x, simd<T1, 8> &x1, const simd<T2, 8> &y,
                  simd<T2, 8> &y1, const simd_mask<8> &m, simd_mask<8> &m1,
                  simd<long long, 32> *out) {
  // simd == simd
  {
    auto v = x == y;
    USE_v;
  }
  // simd == SCALAR
  {
    auto v = x == (T2)5;
    USE_v;
  }
  // SCALAR == simd
  {
    auto v = (T1)5 == y;
    USE_v;
  }

  // mask == mask
  {
    auto v = m == m1;
    USE_v;
  }
  // mask == SCALAR
  {
    auto v = m == 5;
    USE_v;
  }
  // SCALAR == mask
  {
    auto v = 5 == m;
    USE_v;
  }

  // simd_view<simd,...> == simd_view<simd,...>
  {
    auto v = x1.template select<8, 1>() == y1.template select<8, 1>();
    USE_v;
  }
  // simd == simd_view<simd,...>
  {
    auto v = x == y1.template select<8, 1>();
    USE_v;
  }
  // simd_view<simd,...> == simd
  {
    auto v = x1.template select<8, 1>() == y;
    USE_v;
  }

  // simd_view<simd_mask,...> == simd_view<simd_mask,...>
  {
    auto v = m1.select<8, 1>() == m1.select<8, 1>();
    USE_v;
  }
  // simd_mask == simd_view<simd_mask,...>
  {
    auto v = m == m1.select<8, 1>();
    USE_v;
  }
  // simd_view<simd_mask,...> == simd_mask
  {
    auto v = m1.select<8, 1>() == m;
    USE_v;
  }

  // simd_view<simd,...> == SCALAR
  {
    auto v = x1.template select<8, 1>() == (T2)5;
    USE_v;
  }
  // SCALAR == simd_view<simd,...>
  {
    auto v = (T1)5 == y1.template select<8, 1>();
    USE_v;
  }

  // simd_view<simd_mask,...> == SCALAR
  int a = 1;
  {
    auto v = m1.select<8, 1>() == a;
    USE_v;
  }
  // SCALAR == simd_view<simd_mask,...>
  {
    auto v = a == m1.select<8, 1>();
    USE_v;
  }
}

template <class T1, class T2>
[[intel::sycl_explicit_simd]] void equ_cmp_test(simd<long long, 32> *out) {
  simd<T1, 8> x((T1)10);
  simd<T1, 8> x1((T1)11);
  const simd<T2, 8> y((T2)17);
  simd<T2, 8> y1((T2)19);
  const simd_mask<8> m(1);
  simd_mask<8> m1(0);

  equ_cmp_test_impl(x, x1, y, y1, m, m1, out);
}

[[intel::sycl_explicit_simd]] void equ_cmp_tests(simd<long long, 32> *out) {
  equ_cmp_test<unsigned char, long long>(out);
  equ_cmp_test<unsigned char, char>(out);
  equ_cmp_test<int, unsigned short>(out);
  equ_cmp_test<float, char>(out);
  equ_cmp_test<unsigned short, float>(out);
}

// --- comparison

template <class T1, class T2>
[[intel::sycl_explicit_simd]] auto
lt_cmp_test_impl(const simd<T1, 8> &x, simd<T1, 8> &x1, const simd<T2, 8> &y,
                 simd<T2, 8> &y1, const simd_mask<8> &m, simd_mask<8> &m1,
                 simd<long long, 32> *out) {
  // simd < simd
  {
    auto v = x < y;
    USE_v;
  }
  // simd < SCALAR
  {
    auto v = x < (T2)5;
    USE_v;
  }
  // SCALAR < simd
  {
    auto v = (T1)5 == y;
    USE_v;
  }

  // simd_view<simd,...> < simd_view<simd,...>
  {
    auto v = x1.template select<8, 1>() < y1.template select<8, 1>();
    USE_v;
  }
  // simd < simd_view<simd,...>
  {
    auto v = x < y1.template select<8, 1>();
    USE_v;
  }
  // simd_view<simd,...> < simd
  {
    auto v = x1.template select<8, 1>() < y;
    USE_v;
  }

  // simd_view<simd,...> < SCALAR
  {
    auto v = x1.template select<8, 1>() < (T2)5;
    USE_v;
  }
  // SCALAR == simd_view<simd,...>
  {
    auto v = (T1)5 < y1.template select<8, 1>();
    USE_v;
  }
}

template <class T1, class T2>
[[intel::sycl_explicit_simd]] void lt_cmp_test(simd<long long, 32> *out) {
  simd<T1, 8> x((T1)10);
  simd<T1, 8> x1((T1)11);
  const simd<T2, 8> y((T2)17);
  simd<T2, 8> y1((T2)19);
  const simd_mask<8> m(1);
  simd_mask<8> m1(0);

  lt_cmp_test_impl(x, x1, y, y1, m, m1, out);
}

[[intel::sycl_explicit_simd]] void lt_cmp_tests(simd<long long, 32> *out) {
  lt_cmp_test<unsigned char, long long>(out);
  lt_cmp_test<unsigned char, char>(out);
  lt_cmp_test<int, unsigned short>(out);
  lt_cmp_test<float, char>(out);
  lt_cmp_test<unsigned short, float>(out);
}
