// RUN: %clangxx -fsycl -fsyntax-only %s

// This test checks basic APIs of the std::simd and std::simd_mask.

#include <CL/sycl.hpp>
#include <std/experimental/simd.hpp>

#include <type_traits>

using namespace std::experimental;

namespace test {
namespace simd_abi {
template <class T, int N>
using native_fixed_size = typename std::experimental::__simd_abi<
    std::experimental::_StorageKind::_VecExt, N>;
} // namespace simd_abi

template <class T, int N>
using simd = std::experimental::simd<T, simd_abi::native_fixed_size<T, N>>;

template <class T, int N>
using simd_mask =
    std::experimental::simd_mask<T, simd_abi::native_fixed_size<T, N>>;
} // namespace test

using MaskElT = unsigned short;

// Copy constructor
SYCL_EXTERNAL test::simd<int, 8> foo1(test::simd<int, 8> x) { return x; }

SYCL_EXTERNAL test::simd_mask<MaskElT, 8> bar1(test::simd_mask<MaskElT, 8> x) {
  return x;
}

// Broadcast constructor with conversion
SYCL_EXTERNAL test::simd<float, 32> foo2(int x) {
  return test::simd<float, 32>(x);
}

SYCL_EXTERNAL test::simd_mask<MaskElT, 32> bar2(bool x) {
  return test::simd_mask<MaskElT, 32>(x);
}

// Subscript operator
SYCL_EXTERNAL float gee1(test::simd<float, 8> x, int i) { return x[i]; }

SYCL_EXTERNAL bool gee2(test::simd_mask<MaskElT, 8> x, int i) { return x[i]; }

SYCL_EXTERNAL test::simd<int, 8> gee3(test::simd<int, 8> x, int i) {
  x[i] = i;
  return x;
}

SYCL_EXTERNAL test::simd_mask<MaskElT, 8> gee4(test::simd_mask<MaskElT, 8> x,
                                               int i) {
  x[i] = i;
  return x;
}

// Conversion

template <class T, int N>
using clang_vector_t = T __attribute__((ext_vector_type(N)));

SYCL_EXTERNAL clang_vector_t<float, 8> baz1(test::simd<float, 8> x) {
  return x;
}

SYCL_EXTERNAL test::simd<float, 8> baz2(clang_vector_t<float, 8> x) {
  return x;
}
