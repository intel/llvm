// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsyntax-only %s

// This test checks that both host and device compilers can
// successfully compile simd_mask APIs.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

#define DEFINE_BIN_OP_TEST(op, name)                                           \
  template <int N>                                                             \
  SYCL_EXTERNAL SYCL_ESIMD_FUNCTION simd_mask<N> test_impl_##name(             \
      simd_mask<N> &m1, simd_mask<N> &m2) {                                    \
    return m1 op m2;                                                           \
  }                                                                            \
                                                                               \
  simd_mask<1> test_impl_1_##name(simd_mask<1> &m1, simd_mask<1> &m2) {        \
    return test_impl_##name(m1, m2);                                           \
  }                                                                            \
                                                                               \
  simd_mask<17> test_impl_17_##name(simd_mask<17> &m1, simd_mask<17> &m2) {    \
    return test_impl_##name(m1, m2);                                           \
  }                                                                            \
                                                                               \
  simd_mask<32> test_impl_32_##name(simd_mask<32> &m1, simd_mask<32> &m2) {    \
    return test_impl_##name(m1, m2);                                           \
  }

DEFINE_BIN_OP_TEST(&&, and)
DEFINE_BIN_OP_TEST(||, or)
DEFINE_BIN_OP_TEST(&, bit_and)
DEFINE_BIN_OP_TEST(|, bit_or)
DEFINE_BIN_OP_TEST(^, xor)
DEFINE_BIN_OP_TEST(==, eq)
DEFINE_BIN_OP_TEST(!=, ne)
DEFINE_BIN_OP_TEST(&=, bit_and_eq)
DEFINE_BIN_OP_TEST(|=, bit_or_eq)
DEFINE_BIN_OP_TEST(^=, xor_eq)

SYCL_EXTERNAL SYCL_ESIMD_FUNCTION simd_mask<8> misc_tests(bool val) {
  simd_mask<8> m1(val);   // broadcast constructor
  simd_mask<8> m2;        // default constructor
  simd_mask<8> m3(m1[4]); // operator[]
  simd_mask<8> m4 = !m3;  // operator!
  static_assert(m4.length == 8, "size() failed");
  simd<char, 8> ch1(1);
  simd<char, 8> ch2(2);
  simd_mask<8> m5 = ch1 > ch2;
  return m5;
}

SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void compat_test(float *ptr) {
  simd<unsigned short, 16> pred(1);
  simd<unsigned int, 16> offsets;

  // expected-warning@+1 {{deprecated}}
  auto x1 = gather<float, 16>(ptr, offsets, pred);
  // expected-warning@+1 {{deprecated}}
  auto x2 = gather<float, 16>(ptr, offsets, simd<unsigned short, 16>{});
  simd_mask<16> m1(0);
  // expected-warning@+1 {{deprecated}}
  m1 = pred;
  simd_mask<16> m2(0);
  // expected-warning@+1 {{deprecated}}
  m2 = std::move(pred);
}
