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
  if (!res) {
    for (int i = 0; i < Num; ++i) {
      std::cout << "(" << (int)a[i] << " == " << (int)b[i] << ")" << std::endl;
    }
  }
  return res;
}

template <typename T, int Num>
typename std::enable_if<!std::is_same<T, half>::value &&
                            !std::is_same<T, float>::value &&
                            !std::is_same<T, double>::value,
                        void>::type
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
  if constexpr (!std::is_same<T, std::byte>::value) {
    constexpr vec<T, Num> maxs(-std::numeric_limits<T>::infinity());
    assert(compareVectors(known_identity<minimum<>, vec<T, Num>>::value, maxs));
  }

  static_assert(has_known_identity<maximum<>, vec<T, Num>>::value);
  static_assert(has_known_identity<maximum<vec<T, Num>>, vec<T, Num>>::value);
  if constexpr (!std::is_same<T, std::byte>::value) {
    constexpr vec<T, Num> mins(std::numeric_limits<T>::infinity());
    assert(compareVectors(known_identity<maximum<>, vec<T, Num>>::value, mins));
  }
}

template <typename T, int Num>
typename std::enable_if<std::is_same<T, sycl::half>::value ||
                            std::is_same<T, float>::value ||
                            std::is_same<T, double>::value,
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

void checkVecTypesKnownIdentity() {

#define CHECK_VEC(type)                                                        \
  do {                                                                         \
    checkVecKnownIdentity<type, 1>();                                          \
    checkVecKnownIdentity<type, 2>();                                          \
    checkVecKnownIdentity<type, 3>();                                          \
    checkVecKnownIdentity<type, 4>();                                          \
    checkVecKnownIdentity<type, 8>();                                          \
    checkVecKnownIdentity<type, 16>();                                         \
  } while (0)

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  CHECK_VEC(std::byte);
#endif
  CHECK_VEC(int8_t);
  CHECK_VEC(int16_t);
  CHECK_VEC(int32_t);
  CHECK_VEC(int64_t);
  CHECK_VEC(uint8_t);
  CHECK_VEC(uint16_t);
  CHECK_VEC(uint32_t);
  CHECK_VEC(uint64_t);

  CHECK_VEC(char);
  CHECK_VEC(short int);
  CHECK_VEC(int);
  CHECK_VEC(long);
  CHECK_VEC(long long);
  CHECK_VEC(unsigned char);
  CHECK_VEC(unsigned short int);
  CHECK_VEC(unsigned int);
  CHECK_VEC(unsigned long);
  CHECK_VEC(unsigned long long);
  CHECK_VEC(float);
  CHECK_VEC(double);

  checkVecKnownIdentity<half, 2>();
  checkVecKnownIdentity<half, 3>();
  checkVecKnownIdentity<half, 4>();
  checkVecKnownIdentity<half, 8>();
  checkVecKnownIdentity<half, 16>();

#undef CHECK_VEC
}

template <typename T, size_t Num>
bool compareMarrays(const marray<T, Num> a, const marray<T, Num> b) {
  bool res = true;
  for (int i = 0; i < Num; ++i) {
    res &= (a[i] == b[i]);
  }
  if (!res) {
    for (int i = 0; i < Num; ++i) {
      std::cout << "(" << (int)a[i] << " == " << (int)b[i] << ")" << std::endl;
    }
  }
  return res;
}

template <typename T, size_t Num>
typename std::enable_if<!std::is_same<T, half>::value &&
                            !std::is_same<T, float>::value &&
                            !std::is_same<T, double>::value,
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
  if constexpr (!std::is_same<T, std::byte>::value) {
    constexpr marray<T, Num> maxs(-std::numeric_limits<T>::infinity());
    assert(
        compareMarrays(known_identity<minimum<>, marray<T, Num>>::value, maxs));
  }

  static_assert(has_known_identity<maximum<>, marray<T, Num>>::value);
  static_assert(
      has_known_identity<maximum<marray<T, Num>>, marray<T, Num>>::value);
  if constexpr (!std::is_same<T, std::byte>::value) {
    constexpr marray<T, Num> mins(std::numeric_limits<T>::infinity());
    assert(
        compareMarrays(known_identity<maximum<>, marray<T, Num>>::value, mins));
  }
}

template <typename T, int Num>
typename std::enable_if<std::is_same<T, sycl::half>::value ||
                            std::is_same<T, float>::value ||
                            std::is_same<T, double>::value,
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

void checkMarrayTypesKnownIdentity() {

#define CHECK_MARRAY(type)                                                     \
  do {                                                                         \
    checkMarrayKnownIdentity<type, 1>();                                       \
    checkMarrayKnownIdentity<type, 2>();                                       \
    checkMarrayKnownIdentity<type, 3>();                                       \
    checkMarrayKnownIdentity<type, 4>();                                       \
    checkMarrayKnownIdentity<type, 8>();                                       \
    checkMarrayKnownIdentity<type, 16>();                                      \
  } while (0)

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  CHECK_MARRAY(std::byte);
#endif
  CHECK_MARRAY(int8_t);
  CHECK_MARRAY(int16_t);
  CHECK_MARRAY(int32_t);
  CHECK_MARRAY(int64_t);
  CHECK_MARRAY(uint8_t);
  CHECK_MARRAY(uint16_t);
  CHECK_MARRAY(uint32_t);
  CHECK_MARRAY(uint64_t);

  CHECK_MARRAY(char);
  CHECK_MARRAY(short int);
  CHECK_MARRAY(int);
  CHECK_MARRAY(long);
  CHECK_MARRAY(long long);
  CHECK_MARRAY(unsigned char);
  CHECK_MARRAY(unsigned short int);
  CHECK_MARRAY(unsigned int);
  CHECK_MARRAY(unsigned long);
  CHECK_MARRAY(unsigned long long);
  CHECK_MARRAY(half);
  CHECK_MARRAY(float);
  CHECK_MARRAY(double);

#undef CHECK_MARRAY
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
