// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

// This test performs basic checks of has_known_identity and known_identity
// type traits.

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T> void checkCommonBasicKnownIdentity() {
  static_assert(has_known_identity<sycl::maximum<>, T>::value);
  static_assert(has_known_identity<sycl::maximum<T>, T>::value);
  static_assert(has_known_identity<sycl::minimum<>, T>::value);
  static_assert(has_known_identity<sycl::minimum<T>, T>::value);
}

template <typename T> void checkCommonKnownIdentity() {
  checkCommonBasicKnownIdentity<T>();

  static_assert(has_known_identity<std::plus<>, T>::value);
  static_assert(has_known_identity<std::plus<T>, T>::value);
  static_assert(known_identity<std::plus<>, T>::value == 0);
  static_assert(known_identity<std::plus<T>, T>::value == 0);

  static_assert(has_known_identity<std::multiplies<>, T>::value);
  static_assert(has_known_identity<std::multiplies<T>, T>::value);
  static_assert(known_identity<std::multiplies<>, T>::value == 1);
  static_assert(known_identity<std::multiplies<T>, T>::value == 1);
}

template <typename T> void checkIntKnownIdentity() {
  checkCommonKnownIdentity<T>();

  constexpr T Ones = ~static_cast<T>(0);
  static_assert(has_known_identity<std::bit_and<>, T>::value);
  static_assert(has_known_identity<std::bit_and<T>, T>::value);
  static_assert(known_identity<std::bit_and<>, T>::value == Ones);
  static_assert(known_identity<std::bit_and<T>, T>::value == Ones);

  static_assert(has_known_identity<std::bit_or<>, T>::value);
  static_assert(has_known_identity<std::bit_or<T>, T>::value);
  static_assert(known_identity<std::bit_or<>, T>::value == 0);
  static_assert(known_identity<std::bit_or<T>, T>::value == 0);

  static_assert(has_known_identity<std::bit_xor<>, T>::value);
  static_assert(has_known_identity<std::bit_xor<T>, T>::value);
  static_assert(known_identity<std::bit_xor<>, T>::value == 0);
  static_assert(known_identity<std::bit_xor<T>, T>::value == 0);
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

  checkCommonBasicKnownIdentity<half>();
  checkCommonBasicKnownIdentity<sycl::cl_half>();
  checkCommonBasicKnownIdentity<::cl_half>();

  // Few negative tests just to check that it does not always return true.
  static_assert(!has_known_identity<std::minus<>, int>::value);
  static_assert(!has_known_identity<sycl::bit_or<>, float>::value);

  return 0;
}
