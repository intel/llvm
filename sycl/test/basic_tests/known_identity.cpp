// RUN: %clangxx -fsycl -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning -o %t.out -std=c++17
// RUN: %RUN_ON_HOST %t.out
// expected-no-diagnostics

// This test performs basic checks of has_known_identity and known_identity
// type traits.

#include <CL/sycl.hpp>
#include <cassert>
#include <cstddef>

using namespace cl::sycl;

template <typename T> void checkCommonKnownIdentity() {
  static_assert(has_known_identity<sycl::maximum<>, T>::value);
  static_assert(has_known_identity<sycl::maximum<T>, T>::value);
  static_assert(has_known_identity<sycl::minimum<>, T>::value);
  static_assert(has_known_identity<sycl::minimum<T>, T>::value);

  static_assert(has_known_identity<std::plus<>, T>::value);
  static_assert(has_known_identity<std::plus<T>, T>::value);
  static_assert(known_identity<std::plus<>, T>::value == 0);
  static_assert(known_identity<std::plus<T>, T>::value == 0);

  static_assert(has_known_identity<sycl::plus<>, T>::value);
  static_assert(has_known_identity<sycl::plus<T>, T>::value);
  static_assert(known_identity<sycl::plus<>, T>::value == 0);
  static_assert(known_identity<sycl::plus<T>, T>::value == 0);

  static_assert(has_known_identity<std::multiplies<>, T>::value);
  static_assert(has_known_identity<std::multiplies<T>, T>::value);
  static_assert(known_identity<std::multiplies<>, T>::value == 1);
  static_assert(known_identity<std::multiplies<T>, T>::value == 1);

  static_assert(has_known_identity<sycl::multiplies<>, T>::value);
  static_assert(has_known_identity<sycl::multiplies<T>, T>::value);
  static_assert(known_identity<sycl::multiplies<>, T>::value == 1);
  static_assert(known_identity<sycl::multiplies<T>, T>::value == 1);
}

template <typename T> void checkIntKnownIdentity() {
  checkCommonKnownIdentity<T>();

  constexpr T Ones = ~static_cast<T>(0);
  static_assert(has_known_identity<std::bit_and<>, T>::value);
  static_assert(has_known_identity<std::bit_and<T>, T>::value);
  static_assert(known_identity<std::bit_and<>, T>::value == Ones);
  static_assert(known_identity<std::bit_and<T>, T>::value == Ones);

  static_assert(has_known_identity<sycl::bit_and<>, T>::value);
  static_assert(has_known_identity<sycl::bit_and<T>, T>::value);
  static_assert(known_identity<sycl::bit_and<>, T>::value == Ones);
  static_assert(known_identity<sycl::bit_and<T>, T>::value == Ones);

  static_assert(has_known_identity<std::bit_or<>, T>::value);
  static_assert(has_known_identity<std::bit_or<T>, T>::value);
  static_assert(known_identity<std::bit_or<>, T>::value == 0);
  static_assert(known_identity<std::bit_or<T>, T>::value == 0);

  static_assert(has_known_identity<sycl::bit_or<>, T>::value);
  static_assert(has_known_identity<sycl::bit_or<T>, T>::value);
  static_assert(known_identity<sycl::bit_or<>, T>::value == 0);
  static_assert(known_identity<sycl::bit_or<T>, T>::value == 0);

  static_assert(has_known_identity<std::bit_xor<>, T>::value);
  static_assert(has_known_identity<std::bit_xor<T>, T>::value);
  static_assert(known_identity<std::bit_xor<>, T>::value == 0);
  static_assert(known_identity<std::bit_xor<T>, T>::value == 0);

  static_assert(has_known_identity<sycl::bit_xor<>, T>::value);
  static_assert(has_known_identity<sycl::bit_xor<T>, T>::value);
  static_assert(known_identity<sycl::bit_xor<>, T>::value == 0);
  static_assert(known_identity<sycl::bit_xor<T>, T>::value == 0);
}

template <typename T> void checkBoolKnownIdentity() {
  static_assert(has_known_identity<std::logical_and<>, T>::value);
  static_assert(has_known_identity<std::logical_and<T>, T>::value);
  static_assert(known_identity<std::logical_and<>, T>::value == true);
  static_assert(known_identity<std::logical_and<T>, T>::value == true);

  static_assert(has_known_identity<sycl::logical_and<>, T>::value);
  static_assert(has_known_identity<sycl::logical_and<T>, T>::value);
  static_assert(known_identity<sycl::logical_and<>, T>::value == true);
  static_assert(known_identity<sycl::logical_and<T>, T>::value == true);

  static_assert(has_known_identity<std::logical_or<>, T>::value);
  static_assert(has_known_identity<std::logical_or<T>, T>::value);
  static_assert(known_identity<std::logical_or<>, T>::value == false);
  static_assert(known_identity<std::logical_or<T>, T>::value == false);

  static_assert(has_known_identity<sycl::logical_or<>, T>::value);
  static_assert(has_known_identity<sycl::logical_or<T>, T>::value);
  static_assert(known_identity<sycl::logical_or<>, T>::value == false);
  static_assert(known_identity<sycl::logical_or<T>, T>::value == false);
}

template <typename T, int Num>
bool compareVectors(const vec<T, Num> a, const vec<T, Num> b) {
  bool res = true;
  for (int i = 0; i < Num; ++i) {
    res &= (a[i] == b[i]);
  }
  return res;
}

template <typename T, int Num>
std::enable_if_t<!std::is_same_v<T, half> && !std::is_same_v<T, float> &&
                 !std::is_same_v<T, double>>
checkVecKnownIdentity() {
  constexpr vec<T, Num> zeros(T(0));
  constexpr vec<T, Num> ones(T(1));
  constexpr vec<T, Num> bit_ones(~T(0));

  static_assert(has_known_identity<plus<>, vec<T, Num>>::value);
  static_assert(has_known_identity<plus<vec<T, Num>>, vec<T, Num>>::value);
  assert(compareVectors(known_identity<plus<>, vec<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_or<>, vec<T, Num>>::value);
  static_assert(has_known_identity<bit_or<vec<T, Num>>, vec<T, Num>>::value);
  assert(compareVectors(known_identity<bit_or<>, vec<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_xor<>, vec<T, Num>>::value);
  static_assert(has_known_identity<bit_xor<vec<T, Num>>, vec<T, Num>>::value);
  assert(compareVectors(known_identity<bit_xor<>, vec<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_and<>, vec<T, Num>>::value);
  static_assert(has_known_identity<bit_and<vec<T, Num>>, vec<T, Num>>::value);
  assert(
      compareVectors(known_identity<bit_and<>, vec<T, Num>>::value, bit_ones));

  static_assert(has_known_identity<logical_or<>, vec<T, Num>>::value);
  static_assert(
      has_known_identity<logical_or<vec<T, Num>>, vec<T, Num>>::value);
  assert(
      compareVectors(known_identity<logical_or<>, vec<T, Num>>::value, zeros));

  static_assert(has_known_identity<logical_and<>, vec<T, Num>>::value);
  static_assert(
      has_known_identity<logical_and<vec<T, Num>>, vec<T, Num>>::value);
  assert(
      compareVectors(known_identity<logical_and<>, vec<T, Num>>::value, ones));

  static_assert(has_known_identity<multiplies<>, vec<T, Num>>::value);
  static_assert(
      has_known_identity<multiplies<vec<T, Num>>, vec<T, Num>>::value);
  assert(
      compareVectors(known_identity<multiplies<>, vec<T, Num>>::value, ones));

  static_assert(has_known_identity<minimum<>, vec<T, Num>>::value);
  static_assert(has_known_identity<minimum<vec<T, Num>>, vec<T, Num>>::value);
  if constexpr (!std::is_same_v<T, std::byte>) {
    constexpr vec<T, Num> maxs(-std::numeric_limits<T>::infinity());
    assert(compareVectors(known_identity<minimum<>, vec<T, Num>>::value, maxs));
  }

  static_assert(has_known_identity<maximum<>, vec<T, Num>>::value);
  static_assert(has_known_identity<maximum<vec<T, Num>>, vec<T, Num>>::value);
  if constexpr (!std::is_same_v<T, std::byte>) {
    constexpr vec<T, Num> mins(std::numeric_limits<T>::infinity());
    assert(compareVectors(known_identity<maximum<>, vec<T, Num>>::value, mins));
  }
}

template <typename T, int Num>
typename std::enable_if<std::is_same_v<T, sycl::half> ||
                            std::is_same_v<T, float> ||
                            std::is_same_v<T, double>,
                        void>::type
checkVecKnownIdentity() {
  constexpr vec<T, Num> zeros(T(0.0f));
  constexpr vec<T, Num> ones(T(1.0f));

  static_assert(has_known_identity<plus<>, vec<T, Num>>::value);
  static_assert(has_known_identity<plus<vec<T, Num>>, vec<T, Num>>::value);
  assert(compareVectors(known_identity<plus<>, vec<T, Num>>::value, zeros));

  static_assert(has_known_identity<multiplies<>, vec<T, Num>>::value);
  static_assert(
      has_known_identity<multiplies<vec<T, Num>>, vec<T, Num>>::value);
  assert(
      compareVectors(known_identity<multiplies<>, vec<T, Num>>::value, ones));

  static_assert(has_known_identity<minimum<>, vec<T, Num>>::value);
  static_assert(has_known_identity<minimum<vec<T, Num>>, vec<T, Num>>::value);

  static_assert(has_known_identity<maximum<>, vec<T, Num>>::value);
  static_assert(has_known_identity<maximum<vec<T, Num>>, vec<T, Num>>::value);
}

template <typename T> void checkVecTypeKnownIdentity() {
  checkVecKnownIdentity<T, 1>();
  checkVecKnownIdentity<T, 2>();
  checkVecKnownIdentity<T, 3>();
  checkVecKnownIdentity<T, 4>();
  checkVecKnownIdentity<T, 8>();
  checkVecKnownIdentity<T, 16>();
}

void checkVecTypesKnownIdentity() {
#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  checkVecTypeKnownIdentity<std::byte>();
#endif
  checkVecTypeKnownIdentity<int8_t>();
  checkVecTypeKnownIdentity<int16_t>();
  checkVecTypeKnownIdentity<int32_t>();
  checkVecTypeKnownIdentity<int64_t>();
  checkVecTypeKnownIdentity<uint8_t>();
  checkVecTypeKnownIdentity<uint16_t>();
  checkVecTypeKnownIdentity<uint32_t>();
  checkVecTypeKnownIdentity<uint64_t>();

  checkVecTypeKnownIdentity<char>();
  checkVecTypeKnownIdentity<short int>();
  checkVecTypeKnownIdentity<int>();
  checkVecTypeKnownIdentity<long>();
  checkVecTypeKnownIdentity<long long>();
  checkVecTypeKnownIdentity<unsigned char>();
  checkVecTypeKnownIdentity<unsigned short int>();
  checkVecTypeKnownIdentity<unsigned int>();
  checkVecTypeKnownIdentity<unsigned long>();
  checkVecTypeKnownIdentity<unsigned long long>();
  checkVecTypeKnownIdentity<float>();
  checkVecTypeKnownIdentity<double>();

  checkVecKnownIdentity<half, 2>();
  checkVecKnownIdentity<half, 3>();
  checkVecKnownIdentity<half, 4>();
  checkVecKnownIdentity<half, 8>();
  checkVecKnownIdentity<half, 16>();
}

template <typename T, size_t Num>
bool compareMarrays(const marray<T, Num> a, const marray<T, Num> b) {
  bool res = true;
  for (int i = 0; i < Num; ++i) {
    res &= (a[i] == b[i]);
  }
  return res;
}

template <typename T, size_t Num>
typename std::enable_if<!std::is_same_v<T, half> && !std::is_same_v<T, float> &&
                            !std::is_same_v<T, double>,
                        void>::type
checkMarrayKnownIdentity() {
  constexpr marray<T, Num> zeros(T(0));
  constexpr marray<T, Num> ones(T(1));
  constexpr marray<T, Num> bit_ones(~T(0));

  static_assert(has_known_identity<plus<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<plus<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<plus<>, marray<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_or<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<bit_or<marray<T, Num>>, marray<T, Num>>::value);
  assert(
      compareMarrays(known_identity<bit_or<>, marray<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_xor<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<bit_xor<marray<T, Num>>, marray<T, Num>>::value);
  assert(
      compareMarrays(known_identity<bit_xor<>, marray<T, Num>>::value, zeros));

  static_assert(has_known_identity<bit_and<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<bit_and<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<bit_and<>, marray<T, Num>>::value,
                        bit_ones));

  static_assert(has_known_identity<logical_or<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<logical_or<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<logical_or<>, marray<T, Num>>::value,
                        zeros));

  static_assert(has_known_identity<logical_and<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<logical_and<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<logical_and<>, marray<T, Num>>::value,
                        ones));

  static_assert(has_known_identity<multiplies<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<multiplies<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<multiplies<>, marray<T, Num>>::value,
                        ones));

  static_assert(has_known_identity<minimum<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<minimum<marray<T, Num>>, marray<T, Num>>::value);
  if constexpr (!std::is_same_v<T, std::byte>) {
    constexpr marray<T, Num> maxs(-std::numeric_limits<T>::infinity());
    assert(
        compareMarrays(known_identity<minimum<>, marray<T, Num>>::value, maxs));
  }

  static_assert(has_known_identity<maximum<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<maximum<marray<T, Num>>, marray<T, Num>>::value);
  if constexpr (!std::is_same_v<T, std::byte>) {
    constexpr marray<T, Num> mins(std::numeric_limits<T>::infinity());
    assert(
        compareMarrays(known_identity<maximum<>, marray<T, Num>>::value, mins));
  }
}

template <typename T, int Num>
typename std::enable_if<std::is_same_v<T, sycl::half> ||
                            std::is_same_v<T, float> ||
                            std::is_same_v<T, double>,
                        void>::type
checkMarrayKnownIdentity() {
  constexpr marray<T, Num> zeros(T(0.0f));
  constexpr marray<T, Num> ones(T(1.0f));

  static_assert(has_known_identity<plus<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<plus<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<plus<>, marray<T, Num>>::value, zeros));

  static_assert(has_known_identity<multiplies<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<multiplies<marray<T, Num>>, marray<T, Num>>::value);
  assert(compareMarrays(known_identity<multiplies<>, marray<T, Num>>::value,
                        ones));

  static_assert(has_known_identity<minimum<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<minimum<marray<T, Num>>, marray<T, Num>>::value);

  static_assert(has_known_identity<maximum<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<maximum<marray<T, Num>>, marray<T, Num>>::value);
}

template <typename T> void checkMarrayTypeKnownIdentity() {
  checkMarrayKnownIdentity<T, 1>();
  checkMarrayKnownIdentity<T, 2>();
  checkMarrayKnownIdentity<T, 3>();
  checkMarrayKnownIdentity<T, 4>();
  checkMarrayKnownIdentity<T, 8>();
  checkMarrayKnownIdentity<T, 16>();
}

void checkMarrayTypesKnownIdentity() {
#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  checkMarrayTypeKnownIdentity<std::byte>();
#endif
  checkMarrayTypeKnownIdentity<int8_t>();
  checkMarrayTypeKnownIdentity<int16_t>();
  checkMarrayTypeKnownIdentity<int32_t>();
  checkMarrayTypeKnownIdentity<int64_t>();
  checkMarrayTypeKnownIdentity<uint8_t>();
  checkMarrayTypeKnownIdentity<uint16_t>();
  checkMarrayTypeKnownIdentity<uint32_t>();
  checkMarrayTypeKnownIdentity<uint64_t>();

  checkMarrayTypeKnownIdentity<char>();
  checkMarrayTypeKnownIdentity<short int>();
  checkMarrayTypeKnownIdentity<int>();
  checkMarrayTypeKnownIdentity<long>();
  checkMarrayTypeKnownIdentity<long long>();
  checkMarrayTypeKnownIdentity<unsigned char>();
  checkMarrayTypeKnownIdentity<unsigned short int>();
  checkMarrayTypeKnownIdentity<unsigned int>();
  checkMarrayTypeKnownIdentity<unsigned long>();
  checkMarrayTypeKnownIdentity<unsigned long long>();
  checkMarrayTypeKnownIdentity<half>();
  checkMarrayTypeKnownIdentity<float>();
  checkMarrayTypeKnownIdentity<double>();
}

int main() {
  checkIntKnownIdentity<int8_t>();
  checkIntKnownIdentity<char>();
  checkIntKnownIdentity<cl_char>();

  checkIntKnownIdentity<int16_t>();
  checkIntKnownIdentity<short>();
  checkIntKnownIdentity<cl_short>();

  checkIntKnownIdentity<int32_t>();
  checkIntKnownIdentity<int>();
  checkIntKnownIdentity<cl_int>();

  checkIntKnownIdentity<long>();

  checkIntKnownIdentity<int64_t>();
  checkIntKnownIdentity<long long>();
  checkIntKnownIdentity<cl_long>();

  checkIntKnownIdentity<uint8_t>();
  checkIntKnownIdentity<unsigned char>();
  checkIntKnownIdentity<cl_uchar>();

  checkIntKnownIdentity<uint16_t>();
  checkIntKnownIdentity<unsigned short>();
  checkIntKnownIdentity<cl_ushort>();

  checkIntKnownIdentity<uint32_t>();
  checkIntKnownIdentity<unsigned int>();
  checkIntKnownIdentity<unsigned>();
  checkIntKnownIdentity<cl_uint>();

  checkIntKnownIdentity<unsigned long>();

  checkIntKnownIdentity<uint64_t>();
  checkIntKnownIdentity<unsigned long long>();
  checkIntKnownIdentity<cl_ulong>();
  checkIntKnownIdentity<std::size_t>();

  checkCommonKnownIdentity<float>();
  checkCommonKnownIdentity<cl_float>();
  checkCommonKnownIdentity<double>();
  checkCommonKnownIdentity<cl_double>();

  checkCommonKnownIdentity<half>();
  checkCommonKnownIdentity<sycl::cl_half>();
  checkCommonKnownIdentity<::cl_half>();

  checkBoolKnownIdentity<bool>();

  checkVecTypesKnownIdentity();
  checkMarrayTypesKnownIdentity();

  // Few negative tests just to check that it does not always return true.
  static_assert(!has_known_identity<std::minus<>, int>::value);
  static_assert(!has_known_identity<sycl::bit_or<>, float>::value);

  return 0;
}
