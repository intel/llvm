//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/boolean.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

#include <algorithm>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {

__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
template <class T, size_t N> vec<T, 2> to_vec2(marray<T, N> x, size_t start) {
  return {x[start], x[start + 1]};
}
template <class T, size_t N> vec<T, N> to_vec(marray<T, N> x) {
  vec<T, N> vec;
  for (size_t i = 0; i < N; i++)
    vec[i] = x[i];
  return vec;
}
template <class T, int N> marray<T, N> to_marray(vec<T, N> x) {
  marray<T, N> marray;
  for (size_t i = 0; i < N; i++)
    marray[i] = x[i];
  return marray;
}

// Vectors need fixed-size integers instead of fundamental types.
template <typename T, bool Signed, typename Cond = void>
struct same_fixed_size_int;
template <typename T>
struct same_fixed_size_int<T, true, std::enable_if_t<sizeof(T) == 1>> {
  using type = int8_t;
};
template <typename T>
struct same_fixed_size_int<T, true, std::enable_if_t<sizeof(T) == 2>> {
  using type = int16_t;
};
template <typename T>
struct same_fixed_size_int<T, true, std::enable_if_t<sizeof(T) == 4>> {
  using type = int32_t;
};
template <typename T>
struct same_fixed_size_int<T, true, std::enable_if_t<sizeof(T) == 8>> {
  using type = int64_t;
};
template <typename T>
struct same_fixed_size_int<T, false, std::enable_if_t<sizeof(T) == 1>> {
  using type = uint8_t;
};
template <typename T>
struct same_fixed_size_int<T, false, std::enable_if_t<sizeof(T) == 2>> {
  using type = uint16_t;
};
template <typename T>
struct same_fixed_size_int<T, false, std::enable_if_t<sizeof(T) == 4>> {
  using type = uint32_t;
};
template <typename T>
struct same_fixed_size_int<T, false, std::enable_if_t<sizeof(T) == 8>> {
  using type = uint64_t;
};

// Trait for getting an integer type of the same size as T. This propagates
// through vec and marray.
template <typename T, bool Signed, typename Cond = void> struct same_size_int;
template <typename T>
struct same_size_int<
    T, true, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 1>> {
  using type = signed char;
};
template <typename T>
struct same_size_int<
    T, true, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 2>> {
  using type = signed short;
};
template <typename T>
struct same_size_int<
    T, true, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 4>> {
  using type = signed int;
};
template <typename T>
struct same_size_int<
    T, true, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 8>> {
  using type = signed long long;
};
template <typename T>
struct same_size_int<
    T, false, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 1>> {
  using type = unsigned char;
};
template <typename T>
struct same_size_int<
    T, false, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 2>> {
  using type = unsigned short;
};
template <typename T>
struct same_size_int<
    T, false, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 4>> {
  using type = unsigned int;
};
template <typename T>
struct same_size_int<
    T, false, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 8>> {
  using type = unsigned long long;
};
template <typename T, int N, bool Signed>
struct same_size_int<vec<T, N>, Signed> {
  // Use the fixed-size integer types.
  using type = vec<typename same_fixed_size_int<T, Signed>::type, N>;
};
template <typename T, size_t N, bool Signed>
struct same_size_int<marray<T, N>, Signed> {
  using type = marray<typename same_size_int<T, Signed>::type, N>;
};

// Trait for getting a floating point type of the same size as T. This
// propagates through vec and marray.
template <typename T, typename Cond = void> struct same_size_float;
template <typename T>
struct same_size_float<
    T, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 2>> {
  using type = half;
};
template <typename T>
struct same_size_float<
    T, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 4>> {
  using type = float;
};
template <typename T>
struct same_size_float<
    T, std::enable_if_t<!is_marray_or_vec_v<T> && sizeof(T) == 8>> {
  using type = double;
};
template <typename T, int N> struct same_size_float<vec<T, N>> {
  using type = vec<typename same_size_float<T>::type, N>;
};
template <typename T, size_t N> struct same_size_float<marray<T, N>> {
  using type = marray<typename same_size_float<T>::type, N>;
};

template <typename T, bool Signed>
using same_size_int_t = typename same_size_int<T, Signed>::type;
template <typename T>
using same_size_float_t = typename same_size_float<T>::type;
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

#define __SYCL_COMMA ,

#define __SYCL_DEF_BUILTIN_VEC(TYPE)                                           \
  __SYCL_BUILTIN_DEF(TYPE##2)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##3)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##4)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##8)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##16)

#define __SYCL_DEF_BUILTIN_GEOVEC(TYPE)                                        \
  __SYCL_BUILTIN_DEF(TYPE##2)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##3)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##4)

#define __SYCL_DEF_BUILTIN_GEOCROSSVEC(TYPE)                                   \
  __SYCL_BUILTIN_DEF(TYPE##3)                                                  \
  __SYCL_BUILTIN_DEF(TYPE##4)

#define __SYCL_DEF_BUILTIN_GEOMARRAY(TYPE)                                     \
  __SYCL_BUILTIN_DEF(marray<TYPE __SYCL_COMMA 2>)                              \
  __SYCL_BUILTIN_DEF(marray<TYPE __SYCL_COMMA 3>)                              \
  __SYCL_BUILTIN_DEF(marray<TYPE __SYCL_COMMA 4>)

#define __SYCL_DEF_BUILTIN_GEOCROSSMARRAY(TYPE)                                \
  __SYCL_BUILTIN_DEF(marray<TYPE __SYCL_COMMA 3>)                              \
  __SYCL_BUILTIN_DEF(marray<TYPE __SYCL_COMMA 4>)

#define __SYCL_DEF_BUILTIN_MARRAY(TYPE)

#define __SYCL_DEF_BUILTIN_CHAR_SCALAR __SYCL_BUILTIN_DEF(char)
#define __SYCL_DEF_BUILTIN_CHAR_VEC __SYCL_DEF_BUILTIN_VEC(char)
#define __SYCL_DEF_BUILTIN_CHAR_MARRAY __SYCL_DEF_BUILTIN_MARRAY(char)
#define __SYCL_DEF_BUILTIN_CHARN                                               \
  __SYCL_DEF_BUILTIN_CHAR_VEC                                                  \
  __SYCL_DEF_BUILTIN_CHAR_MARRAY
#define __SYCL_DEF_BUILTIN_SCHAR_SCALAR __SYCL_BUILTIN_DEF(signed char)
#define __SYCL_DEF_BUILTIN_SCHAR_VEC __SYCL_DEF_BUILTIN_VEC(schar)
#define __SYCL_DEF_BUILTIN_SCHAR_MARRAY __SYCL_DEF_BUILTIN_MARRAY(signed char)
#define __SYCL_DEF_BUILTIN_SCHARN                                              \
  __SYCL_DEF_BUILTIN_SCHAR_VEC                                                 \
  __SYCL_DEF_BUILTIN_SCHAR_MARRAY
#define __SYCL_DEF_BUILTIN_IGENCHAR                                            \
  __SYCL_DEF_BUILTIN_SCHAR_SCALAR                                              \
  __SYCL_DEF_BUILTIN_SCHARN
#define __SYCL_DEF_BUILTIN_UCHAR_SCALAR __SYCL_BUILTIN_DEF(unsigned char)
#define __SYCL_DEF_BUILTIN_UCHAR_VEC __SYCL_DEF_BUILTIN_VEC(uchar)
#define __SYCL_DEF_BUILTIN_UCHAR_MARRAY __SYCL_DEF_BUILTIN_MARRAY(unsigned char)
#define __SYCL_DEF_BUILTIN_UCHARN                                              \
  __SYCL_DEF_BUILTIN_UCHAR_VEC                                                 \
  __SYCL_DEF_BUILTIN_UCHAR_MARRAY
#define __SYCL_DEF_BUILTIN_UGENCHAR                                            \
  __SYCL_DEF_BUILTIN_UCHAR_SCALAR                                              \
  __SYCL_DEF_BUILTIN_UCHARN
// schar{n} and char{n} have the same type, so we skip the char{n} variants.
#define __SYCL_DEF_BUILTIN_GENCHAR                                             \
  __SYCL_DEF_BUILTIN_CHAR_SCALAR                                               \
  __SYCL_DEF_BUILTIN_CHAR_MARRAY                                               \
  __SYCL_DEF_BUILTIN_IGENCHAR                                                  \
  __SYCL_DEF_BUILTIN_UGENCHAR

#define __SYCL_DEF_BUILTIN_SHORT_SCALAR __SYCL_BUILTIN_DEF(short)
#define __SYCL_DEF_BUILTIN_SHORT_VEC __SYCL_DEF_BUILTIN_VEC(short)
#define __SYCL_DEF_BUILTIN_SHORT_MARRAY __SYCL_DEF_BUILTIN_MARRAY(short)
#define __SYCL_DEF_BUILTIN_SHORTN                                              \
  __SYCL_DEF_BUILTIN_SHORT_VEC                                                 \
  __SYCL_DEF_BUILTIN_SHORT_MARRAY
#define __SYCL_DEF_BUILTIN_GENSHORT                                            \
  __SYCL_DEF_BUILTIN_SHORT_SCALAR                                              \
  __SYCL_DEF_BUILTIN_SHORTN
#define __SYCL_DEF_BUILTIN_USHORT_SCALAR __SYCL_BUILTIN_DEF(unsigned short)
#define __SYCL_DEF_BUILTIN_USHORT_VEC __SYCL_DEF_BUILTIN_VEC(ushort)
#define __SYCL_DEF_BUILTIN_USHORT_MARRAY                                       \
  __SYCL_DEF_BUILTIN_MARRAY(unsigned short)
#define __SYCL_DEF_BUILTIN_USHORTN                                             \
  __SYCL_DEF_BUILTIN_USHORT_VEC                                                \
  __SYCL_DEF_BUILTIN_USHORT_MARRAY
#define __SYCL_DEF_BUILTIN_UGENSHORT                                           \
  __SYCL_DEF_BUILTIN_USHORT_SCALAR                                             \
  __SYCL_DEF_BUILTIN_USHORTN

#define __SYCL_DEF_BUILTIN_INT_SCALAR __SYCL_BUILTIN_DEF(int)
#define __SYCL_DEF_BUILTIN_INT_VEC __SYCL_DEF_BUILTIN_VEC(int)
#define __SYCL_DEF_BUILTIN_INT_MARRAY __SYCL_DEF_BUILTIN_MARRAY(int)
#define __SYCL_DEF_BUILTIN_INTN                                                \
  __SYCL_DEF_BUILTIN_INT_VEC                                                   \
  __SYCL_DEF_BUILTIN_INT_MARRAY
#define __SYCL_DEF_BUILTIN_GENINT                                              \
  __SYCL_DEF_BUILTIN_INT_SCALAR                                                \
  __SYCL_DEF_BUILTIN_INTN
#define __SYCL_DEF_BUILTIN_UINT_SCALAR __SYCL_BUILTIN_DEF(unsigned int)
#define __SYCL_DEF_BUILTIN_UINT_VEC __SYCL_DEF_BUILTIN_VEC(uint)
#define __SYCL_DEF_BUILTIN_UINT_MARRAY __SYCL_DEF_BUILTIN_MARRAY(unsigned int)
#define __SYCL_DEF_BUILTIN_UINTN                                               \
  __SYCL_DEF_BUILTIN_UINT_VEC                                                  \
  __SYCL_DEF_BUILTIN_UINT_MARRAY
#define __SYCL_DEF_BUILTIN_UGENINT                                             \
  __SYCL_DEF_BUILTIN_UINT_SCALAR                                               \
  __SYCL_DEF_BUILTIN_UINTN

#define __SYCL_DEF_BUILTIN_LONG_SCALAR __SYCL_BUILTIN_DEF(long)
#define __SYCL_DEF_BUILTIN_LONG_VEC __SYCL_DEF_BUILTIN_VEC(long)
#define __SYCL_DEF_BUILTIN_LONG_MARRAY __SYCL_DEF_BUILTIN_MARRAY(long)
#define __SYCL_DEF_BUILTIN_LONGN                                               \
  __SYCL_DEF_BUILTIN_LONG_VEC                                                  \
  __SYCL_DEF_BUILTIN_LONG_MARRAY
#define __SYCL_DEF_BUILTIN_GENLONG                                             \
  __SYCL_DEF_BUILTIN_LONG_SCALAR                                               \
  __SYCL_DEF_BUILTIN_LONGN
#define __SYCL_DEF_BUILTIN_ULONG_SCALAR __SYCL_BUILTIN_DEF(unsigned long)
#define __SYCL_DEF_BUILTIN_ULONG_VEC __SYCL_DEF_BUILTIN_VEC(ulong)
#define __SYCL_DEF_BUILTIN_ULONG_MARRAY __SYCL_DEF_BUILTIN_MARRAY(unsigned long)
#define __SYCL_DEF_BUILTIN_ULONGN                                              \
  __SYCL_DEF_BUILTIN_ULONG_VEC                                                 \
  __SYCL_DEF_BUILTIN_ULONG_MARRAY
#define __SYCL_DEF_BUILTIN_UGENLONG                                            \
  __SYCL_DEF_BUILTIN_ULONG_SCALAR                                              \
  __SYCL_DEF_BUILTIN_ULONGN

#define __SYCL_DEF_BUILTIN_LONGLONG_SCALAR __SYCL_BUILTIN_DEF(long long)
#define __SYCL_DEF_BUILTIN_LONGLONG_VEC __SYCL_DEF_BUILTIN_VEC(longlong)
#define __SYCL_DEF_BUILTIN_LONGLONG_MARRAY __SYCL_DEF_BUILTIN_MARRAY(long long)
#define __SYCL_DEF_BUILTIN_LONGLONGN                                           \
  __SYCL_DEF_BUILTIN_LONGLONG_VEC                                              \
  __SYCL_DEF_BUILTIN_LONGLONG_MARRAY
#define __SYCL_DEF_BUILTIN_GENLONGLONG                                         \
  __SYCL_DEF_BUILTIN_LONGLONG_SCALAR                                           \
  __SYCL_DEF_BUILTIN_LONGLONGN
#define __SYCL_DEF_BUILTIN_ULONGLONG_SCALAR                                    \
  __SYCL_BUILTIN_DEF(unsigned long long)
#define __SYCL_DEF_BUILTIN_ULONGLONG_VEC __SYCL_DEF_BUILTIN_VEC(ulonglong)
#define __SYCL_DEF_BUILTIN_ULONGLONG_MARRAY                                    \
  __SYCL_DEF_BUILTIN_MARRAY(unsigned long long)
#define __SYCL_DEF_BUILTIN_ULONGLONGN                                          \
  __SYCL_DEF_BUILTIN_ULONGLONG_VEC                                             \
  __SYCL_DEF_BUILTIN_ULONGLONG_MARRAY
#define __SYCL_DEF_BUILTIN_UGENLONGLONG                                        \
  __SYCL_DEF_BUILTIN_ULONGLONG_SCALAR                                          \
  __SYCL_DEF_BUILTIN_ULONGLONGN

// longlongn and long{n} have the same types, so we only include one here.
#define __SYCL_DEF_BUILTIN_IGENLONGINTEGER                                     \
  __SYCL_DEF_BUILTIN_LONG_SCALAR                                               \
  __SYCL_DEF_BUILTIN_LONG_MARRAY                                               \
  __SYCL_DEF_BUILTIN_LONGLONG_SCALAR                                           \
  __SYCL_DEF_BUILTIN_LONGLONG_MARRAY                                           \
  __SYCL_DEF_BUILTIN_LONG_VEC

// longlong{n} and long{n} have the same types, so we only include one here.
#define __SYCL_DEF_BUILTIN_UGENLONGINTEGER                                     \
  __SYCL_DEF_BUILTIN_ULONG_SCALAR                                              \
  __SYCL_DEF_BUILTIN_ULONG_MARRAY                                              \
  __SYCL_DEF_BUILTIN_ULONGLONG_SCALAR                                          \
  __SYCL_DEF_BUILTIN_ULONGLONG_MARRAY                                          \
  __SYCL_DEF_BUILTIN_ULONG_VEC

#define __SYCL_DEF_BUILTIN_SIGENINTEGER                                        \
  __SYCL_DEF_BUILTIN_SCHAR_SCALAR                                              \
  __SYCL_DEF_BUILTIN_SHORT_SCALAR                                              \
  __SYCL_DEF_BUILTIN_INT_SCALAR                                                \
  __SYCL_DEF_BUILTIN_LONG_SCALAR                                               \
  __SYCL_DEF_BUILTIN_LONGLONG_SCALAR

// longlongn and longn have the same types, so we only include one here.
#define __SYCL_DEF_BUILTIN_VIGENINTEGER                                        \
  __SYCL_DEF_BUILTIN_CHAR_VEC                                                  \
  __SYCL_DEF_BUILTIN_SHORT_VEC                                                 \
  __SYCL_DEF_BUILTIN_INT_VEC                                                   \
  __SYCL_DEF_BUILTIN_LONG_VEC

#define __SYCL_DEF_BUILTIN_IGENINTEGER                                         \
  __SYCL_DEF_BUILTIN_IGENCHAR                                                  \
  __SYCL_DEF_BUILTIN_GENSHORT                                                  \
  __SYCL_DEF_BUILTIN_GENINT                                                    \
  __SYCL_DEF_BUILTIN_IGENLONGINTEGER

#define __SYCL_DEF_BUILTIN_SUGENINTEGER                                        \
  __SYCL_DEF_BUILTIN_UCHAR_SCALAR                                              \
  __SYCL_DEF_BUILTIN_USHORT_SCALAR                                             \
  __SYCL_DEF_BUILTIN_UINT_SCALAR                                               \
  __SYCL_DEF_BUILTIN_ULONG_SCALAR                                              \
  __SYCL_DEF_BUILTIN_ULONGLONG_SCALAR

// longlongn and longn have the same types, so we only include one here.
#define __SYCL_DEF_BUILTIN_VUGENINTEGER                                        \
  __SYCL_DEF_BUILTIN_UCHAR_VEC                                                 \
  __SYCL_DEF_BUILTIN_USHORT_VEC                                                \
  __SYCL_DEF_BUILTIN_UINT_VEC                                                  \
  __SYCL_DEF_BUILTIN_ULONG_VEC

#define __SYCL_DEF_BUILTIN_UGENINTEGER                                         \
  __SYCL_DEF_BUILTIN_UGENCHAR                                                  \
  __SYCL_DEF_BUILTIN_UGENSHORT                                                 \
  __SYCL_DEF_BUILTIN_UGENINT                                                   \
  __SYCL_DEF_BUILTIN_UGENLONGINTEGER

#define __SYCL_DEF_BUILTIN_SGENINTEGER                                         \
  __SYCL_DEF_BUILTIN_CHAR_SCALAR                                               \
  __SYCL_DEF_BUILTIN_SIGENINTEGER                                              \
  __SYCL_DEF_BUILTIN_SUGENINTEGER

// longlongn and long{n} have the same types, so we only include one here.
#define __SYCL_DEF_BUILTIN_VGENINTEGER                                         \
  __SYCL_DEF_BUILTIN_CHAR_VEC                                                  \
  __SYCL_DEF_BUILTIN_UCHAR_VEC                                                 \
  __SYCL_DEF_BUILTIN_SHORT_VEC                                                 \
  __SYCL_DEF_BUILTIN_USHORT_VEC                                                \
  __SYCL_DEF_BUILTIN_INT_VEC                                                   \
  __SYCL_DEF_BUILTIN_UINT_VEC                                                  \
  __SYCL_DEF_BUILTIN_LONG_VEC                                                  \
  __SYCL_DEF_BUILTIN_ULONG_VEC

#define __SYCL_DEF_BUILTIN_GENINTEGER                                          \
  __SYCL_DEF_BUILTIN_GENCHAR                                                   \
  __SYCL_DEF_BUILTIN_GENSHORT                                                  \
  __SYCL_DEF_BUILTIN_UGENSHORT                                                 \
  __SYCL_DEF_BUILTIN_GENINT                                                    \
  __SYCL_DEF_BUILTIN_UGENINT                                                   \
  __SYCL_DEF_BUILTIN_UGENLONGINTEGER                                           \
  __SYCL_DEF_BUILTIN_IGENLONGINTEGER

#define __SYCL_DEF_BUILTIN_FLOAT_SCALAR __SYCL_BUILTIN_DEF(float)
#define __SYCL_DEF_BUILTIN_FLOAT_VEC __SYCL_DEF_BUILTIN_VEC(float)
#define __SYCL_DEF_BUILTIN_FLOAT_GEOVEC __SYCL_DEF_BUILTIN_GEOVEC(float)
#define __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSMARRAY                                \
  __SYCL_DEF_BUILTIN_GEOCROSSMARRAY(float)
#define __SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY __SYCL_DEF_BUILTIN_GEOMARRAY(float)
#define __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSVEC                                   \
  __SYCL_DEF_BUILTIN_GEOCROSSVEC(float)
#define __SYCL_DEF_BUILTIN_FLOAT_MARRAY __SYCL_DEF_BUILTIN_MARRAY(float)
#define __SYCL_DEF_BUILTIN_FLOATN                                              \
  __SYCL_DEF_BUILTIN_FLOAT_VEC                                                 \
  __SYCL_DEF_BUILTIN_FLOAT_MARRAY
#define __SYCL_DEF_BUILTIN_GENFLOATF                                           \
  __SYCL_DEF_BUILTIN_FLOAT_SCALAR                                              \
  __SYCL_DEF_BUILTIN_FLOATN
#define __SYCL_DEF_BUILTIN_GENGEOFLOATF                                        \
  __SYCL_DEF_BUILTIN_FLOAT_SCALAR                                              \
  __SYCL_DEF_BUILTIN_FLOAT_GEOVEC

#define __SYCL_DEF_BUILTIN_DOUBLE_SCALAR __SYCL_BUILTIN_DEF(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_VEC __SYCL_DEF_BUILTIN_VEC(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_GEOVEC __SYCL_DEF_BUILTIN_GEOVEC(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSMARRAY                               \
  __SYCL_DEF_BUILTIN_GEOCROSSMARRAY(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_GEOMARRAY __SYCL_DEF_BUILTIN_GEOMARRAY(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSVEC                                  \
  __SYCL_DEF_BUILTIN_GEOCROSSVEC(double)
#define __SYCL_DEF_BUILTIN_DOUBLE_MARRAY __SYCL_DEF_BUILTIN_MARRAY(double)
#define __SYCL_DEF_BUILTIN_DOUBLEN                                             \
  __SYCL_DEF_BUILTIN_DOUBLE_VEC                                                \
  __SYCL_DEF_BUILTIN_DOUBLE_MARRAY
#define __SYCL_DEF_BUILTIN_GENFLOATD                                           \
  __SYCL_DEF_BUILTIN_DOUBLE_SCALAR                                             \
  __SYCL_DEF_BUILTIN_DOUBLEN
#define __SYCL_DEF_BUILTIN_GENGEOFLOATD                                        \
  __SYCL_DEF_BUILTIN_DOUBLE_SCALAR                                             \
  __SYCL_DEF_BUILTIN_DOUBLE_GEOVEC

#define __SYCL_DEF_BUILTIN_HALF_SCALAR __SYCL_BUILTIN_DEF(half)
#define __SYCL_DEF_BUILTIN_HALF_VEC __SYCL_DEF_BUILTIN_VEC(half)
#define __SYCL_DEF_BUILTIN_HALF_GEOVEC __SYCL_DEF_BUILTIN_GEOVEC(half)
#define __SYCL_DEF_BUILTIN_HALF_GEOCROSSMARRAY                                 \
  __SYCL_DEF_BUILTIN_GEOCROSSMARRAY(half)
#define __SYCL_DEF_BUILTIN_HALF_GEOMARRAY __SYCL_DEF_BUILTIN_GEOMARRAY(half)
#define __SYCL_DEF_BUILTIN_HALF_GEOCROSSVEC __SYCL_DEF_BUILTIN_GEOCROSSVEC(half)
#define __SYCL_DEF_BUILTIN_HALF_MARRAY __SYCL_DEF_BUILTIN_MARRAY(half)
#define __SYCL_DEF_BUILTIN_HALFN                                               \
  __SYCL_DEF_BUILTIN_HALF_VEC                                                  \
  __SYCL_DEF_BUILTIN_HALF_MARRAY
#define __SYCL_DEF_BUILTIN_GENFLOATH                                           \
  __SYCL_DEF_BUILTIN_HALF_SCALAR                                               \
  __SYCL_DEF_BUILTIN_HALFN
#define __SYCL_DEF_BUILTIN_GENGEOFLOATH                                        \
  __SYCL_DEF_BUILTIN_HALF_SCALAR                                               \
  __SYCL_DEF_BUILTIN_HALF_GEOVEC

#define __SYCL_DEF_BUILTIN_SGENFLOAT                                           \
  __SYCL_DEF_BUILTIN_FLOAT_SCALAR                                              \
  __SYCL_DEF_BUILTIN_DOUBLE_SCALAR                                             \
  __SYCL_DEF_BUILTIN_HALF_SCALAR

#define __SYCL_DEF_BUILTIN_VGENFLOAT                                           \
  __SYCL_DEF_BUILTIN_FLOAT_VEC                                                 \
  __SYCL_DEF_BUILTIN_DOUBLE_VEC                                                \
  __SYCL_DEF_BUILTIN_HALF_VEC

#define __SYCL_DEF_BUILTIN_GENFLOAT                                            \
  __SYCL_DEF_BUILTIN_GENFLOATF                                                 \
  __SYCL_DEF_BUILTIN_GENFLOATD                                                 \
  __SYCL_DEF_BUILTIN_GENFLOATH

#define __SYCL_DEF_BUILTIN_GENGEOFLOAT                                         \
  __SYCL_DEF_BUILTIN_GENGEOFLOATF                                              \
  __SYCL_DEF_BUILTIN_GENGEOFLOATD                                              \
  __SYCL_DEF_BUILTIN_GENGEOFLOATH

#define __SYCL_DEF_BUILTIN_GENGEOCROSSMARRAY                                   \
  __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSMARRAY                                      \
  __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSMARRAY                                     \
  __SYCL_DEF_BUILTIN_HALF_GEOCROSSMARRAY

#define __SYCL_DEF_BUILTIN_GENGEOMARRAY                                        \
  __SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY                                           \
  __SYCL_DEF_BUILTIN_DOUBLE_GEOMARRAY                                          \
  __SYCL_DEF_BUILTIN_HALF_GEOMARRAY

// TODO: Replace with overloads.
#define __SYCL_DEF_BUILTIN_VGENGEOCROSSFLOAT                                   \
  __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSVEC                                         \
  __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSVEC                                        \
  __SYCL_DEF_BUILTIN_HALF_GEOCROSSVEC

#define __SYCL_DEF_BUILTIN_VGENGEOFLOAT                                        \
  __SYCL_DEF_BUILTIN_FLOAT_GEOVEC                                              \
  __SYCL_DEF_BUILTIN_DOUBLE_GEOVEC                                             \
  __SYCL_DEF_BUILTIN_HALF_GEOVEC

// TODO: Replace with overloads.
#ifdef __FAST_MATH__
#define __FAST_MATH_SGENFLOAT(T)                                               \
  (std::is_same_v<T, double> || std::is_same_v<T, half>)
#else
#define __FAST_MATH_SGENFLOAT(T) (detail::is_sgenfloat<T>::value)
#endif

#ifdef __FAST_MATH__
#define __SYCL_DEF_BUILTIN_FAST_MATH_GENFLOAT                                  \
  __SYCL_DEF_BUILTIN_GENFLOATD                                                 \
  __SYCL_DEF_BUILTIN_GENFLOATH
#else
#define __SYCL_DEF_BUILTIN_FAST_MATH_GENFLOAT __SYCL_DEF_BUILTIN_GENFLOAT
#endif

#define __SYCL_DEF_BUILTIN_SGENTYPE                                            \
  __SYCL_DEF_BUILTIN_SGENINTEGER                                               \
  __SYCL_DEF_BUILTIN_SGENFLOAT

#define __SYCL_DEF_BUILTIN_VGENTYPE                                            \
  __SYCL_DEF_BUILTIN_VGENINTEGER                                               \
  __SYCL_DEF_BUILTIN_VGENFLOAT

#define __SYCL_DEF_BUILTIN_GENTYPE                                             \
  __SYCL_DEF_BUILTIN_GENINTEGER                                                \
  __SYCL_DEF_BUILTIN_GENFLOAT

/* ----------------- 4.13.3 Math functions. ---------------------------------*/

// TODO: Replace with overloads.
// These macros for marray math function implementations use vectorizations of
// size two as a simple general optimization. A more complex implementation
// using larger vectorizations for large marray sizes is possible; however more
// testing is required in order to ascertain the performance implications for
// all backends.
#define __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                               \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N / 2; i++) {                                         \
    vec<T, 2> partial_res =                                                    \
        __sycl_std::__invoke_##NAME<vec<T, 2>>(detail::to_vec2(x, i * 2));     \
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));                 \
  }                                                                            \
  if (N % 2) {                                                                 \
    res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1]);                     \
  }                                                                            \
  return res;

#define __SYCL_MATH_FUNCTION_OVERLOAD(NAME)                                    \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x) __NOEXC {                                           \
    __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                                   \
  }

__SYCL_MATH_FUNCTION_OVERLOAD(cospi)
__SYCL_MATH_FUNCTION_OVERLOAD(sinpi)
__SYCL_MATH_FUNCTION_OVERLOAD(tanpi)
__SYCL_MATH_FUNCTION_OVERLOAD(sinh)
__SYCL_MATH_FUNCTION_OVERLOAD(cosh)
__SYCL_MATH_FUNCTION_OVERLOAD(tanh)
__SYCL_MATH_FUNCTION_OVERLOAD(asin)
__SYCL_MATH_FUNCTION_OVERLOAD(acos)
__SYCL_MATH_FUNCTION_OVERLOAD(atan)
__SYCL_MATH_FUNCTION_OVERLOAD(asinpi)
__SYCL_MATH_FUNCTION_OVERLOAD(acospi)
__SYCL_MATH_FUNCTION_OVERLOAD(atanpi)
__SYCL_MATH_FUNCTION_OVERLOAD(asinh)
__SYCL_MATH_FUNCTION_OVERLOAD(acosh)
__SYCL_MATH_FUNCTION_OVERLOAD(atanh)
__SYCL_MATH_FUNCTION_OVERLOAD(cbrt)
__SYCL_MATH_FUNCTION_OVERLOAD(ceil)
__SYCL_MATH_FUNCTION_OVERLOAD(floor)
__SYCL_MATH_FUNCTION_OVERLOAD(erfc)
__SYCL_MATH_FUNCTION_OVERLOAD(erf)
__SYCL_MATH_FUNCTION_OVERLOAD(expm1)
__SYCL_MATH_FUNCTION_OVERLOAD(tgamma)
__SYCL_MATH_FUNCTION_OVERLOAD(lgamma)
__SYCL_MATH_FUNCTION_OVERLOAD(log1p)
__SYCL_MATH_FUNCTION_OVERLOAD(logb)
__SYCL_MATH_FUNCTION_OVERLOAD(rint)
__SYCL_MATH_FUNCTION_OVERLOAD(round)
__SYCL_MATH_FUNCTION_OVERLOAD(trunc)
__SYCL_MATH_FUNCTION_OVERLOAD(fabs)

#undef __SYCL_MATH_FUNCTION_OVERLOAD

// __SYCL_MATH_FUNCTION_OVERLOAD_FM cases are replaced by corresponding native
// implementations when the -ffast-math flag is used with float.
#define __SYCL_MATH_FUNCTION_OVERLOAD_FM(NAME)                                 \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<__FAST_MATH_SGENFLOAT(T), marray<T, N>>                 \
      NAME(marray<T, N> x) __NOEXC {                                           \
    __SYCL_MATH_FUNCTION_OVERLOAD_IMPL(NAME)                                   \
  }

__SYCL_MATH_FUNCTION_OVERLOAD_FM(sin)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(cos)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(tan)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(sqrt)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(rsqrt)

#undef __SYCL_MATH_FUNCTION_OVERLOAD_FM
#undef __SYCL_MATH_FUNCTION_OVERLOAD_IMPL

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<int, N>>
    ilogb(marray<T, N> x) __NOEXC {
  marray<int, N> res;
  for (size_t i = 0; i < N / 2; i++) {
    vec<int, 2> partial_res =
        __sycl_std::__invoke_ilogb<vec<int, 2>>(detail::to_vec2(x, i * 2));
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<int, 2>));
  }
  if (N % 2) {
    res[N - 1] = __sycl_std::__invoke_ilogb<int>(x[N - 1]);
  }
  return res;
}

#define __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(NAME)                             \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N / 2; i++) {                                         \
    auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(                 \
        detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));                 \
    std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));                 \
  }                                                                            \
  if (N % 2) {                                                                 \
    res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1], y[N - 1]);           \
  }                                                                            \
  return res;

#define __SYCL_MATH_FUNCTION_2_OVERLOAD(NAME)                                  \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, marray<T, N> y) __NOEXC {                           \
    __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(NAME)                                 \
  }

__SYCL_MATH_FUNCTION_2_OVERLOAD(atan2)
__SYCL_MATH_FUNCTION_2_OVERLOAD(atan2pi)
__SYCL_MATH_FUNCTION_2_OVERLOAD(copysign)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fdim)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmin)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmax)
__SYCL_MATH_FUNCTION_2_OVERLOAD(fmod)
__SYCL_MATH_FUNCTION_2_OVERLOAD(hypot)
__SYCL_MATH_FUNCTION_2_OVERLOAD(maxmag)
__SYCL_MATH_FUNCTION_2_OVERLOAD(minmag)
__SYCL_MATH_FUNCTION_2_OVERLOAD(nextafter)
__SYCL_MATH_FUNCTION_2_OVERLOAD(pow)
__SYCL_MATH_FUNCTION_2_OVERLOAD(remainder)

#undef __SYCL_MATH_FUNCTION_2_OVERLOAD

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<__FAST_MATH_SGENFLOAT(T), marray<T, N>>
    powr(marray<T, N> x,
         marray<T, N> y) __NOEXC{__SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL(powr)}

#undef __SYCL_MATH_FUNCTION_2_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(NAME)                      \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, T y) __NOEXC {                                      \
    marray<T, N> res;                                                          \
    sycl::vec<T, 2> y_vec{y, y};                                               \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(               \
          detail::to_vec2(x, i * 2), y_vec);                                   \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));               \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_##NAME<T>(x[N - 1], y_vec[0]);         \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(fmax)
    // clang-format off
__SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD(fmin)

#undef __SYCL_MATH_FUNCTION_2_SGENFLOAT_Y_OVERLOAD

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    ldexp(marray<T, N> x, marray<int, N> k) __NOEXC {
  // clang-format on
  marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = __sycl_std::__invoke_ldexp<T>(x[i], k[i]);
  }
  return res;
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    ldexp(marray<T, N> x, int k) __NOEXC {
  marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = __sycl_std::__invoke_ldexp<T>(x[i], k);
  }
  return res;
}

#define __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(NAME)                    \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N; i++) {                                             \
    res[i] = __sycl_std::__invoke_##NAME<T>(x[i], y[i]);                       \
  }                                                                            \
  return res;

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    pown(marray<T, N> x, marray<int, N> y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    rootn(marray<T, N> x, marray<int, N> y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL(rootn)
}

#undef __SYCL_MATH_FUNCTION_2_GENINT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(NAME)                       \
  marray<T, N> res;                                                            \
  for (size_t i = 0; i < N; i++) {                                             \
    res[i] = __sycl_std::__invoke_##NAME<T>(x[i], y);                          \
  }                                                                            \
  return res;

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    pown(marray<T, N> x, int y) __NOEXC {
  __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(pown)
}

template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>
    rootn(marray<T, N> x,
          int y) __NOEXC{__SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL(rootn)}

#undef __SYCL_MATH_FUNCTION_2_INT_Y_OVERLOAD_IMPL

#define __SYCL_MATH_FUNCTION_3_OVERLOAD(NAME)                                  \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<detail::is_sgenfloat<T>::value, marray<T, N>>           \
      NAME(marray<T, N> x, marray<T, N> y, marray<T, N> z) __NOEXC {           \
    marray<T, N> res;                                                          \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_##NAME<vec<T, 2>>(               \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2),                \
          detail::to_vec2(z, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<T, 2>));               \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_##NAME<T>(x[N - 1], y[N - 1], z[N - 1]);        \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MATH_FUNCTION_3_OVERLOAD(mad) __SYCL_MATH_FUNCTION_3_OVERLOAD(mix)
    __SYCL_MATH_FUNCTION_3_OVERLOAD(fma)

#undef __SYCL_MATH_FUNCTION_3_OVERLOAD

    // genfloat acos (genfloat x)
    inline float acos(float x) __NOEXC {
  return __sycl_std::__invoke_acos<float>(x);
}
inline float2 acos(float2 x) __NOEXC {
  return __sycl_std::__invoke_acos<float2>(x);
}
inline float3 acos(float3 x) __NOEXC {
  return __sycl_std::__invoke_acos<float3>(x);
}
inline float4 acos(float4 x) __NOEXC {
  return __sycl_std::__invoke_acos<float4>(x);
}
inline float8 acos(float8 x) __NOEXC {
  return __sycl_std::__invoke_acos<float8>(x);
}
inline float16 acos(float16 x) __NOEXC {
  return __sycl_std::__invoke_acos<float16>(x);
}
inline double acos(double x) __NOEXC {
  return __sycl_std::__invoke_acos<double>(x);
}
inline double2 acos(double2 x) __NOEXC {
  return __sycl_std::__invoke_acos<double2>(x);
}
inline double3 acos(double3 x) __NOEXC {
  return __sycl_std::__invoke_acos<double3>(x);
}
inline double4 acos(double4 x) __NOEXC {
  return __sycl_std::__invoke_acos<double4>(x);
}
inline double8 acos(double8 x) __NOEXC {
  return __sycl_std::__invoke_acos<double8>(x);
}
inline double16 acos(double16 x) __NOEXC {
  return __sycl_std::__invoke_acos<double16>(x);
}
inline half acos(half x) __NOEXC { return __sycl_std::__invoke_acos<half>(x); }
inline half2 acos(half2 x) __NOEXC {
  return __sycl_std::__invoke_acos<half2>(x);
}
inline half3 acos(half3 x) __NOEXC {
  return __sycl_std::__invoke_acos<half3>(x);
}
inline half4 acos(half4 x) __NOEXC {
  return __sycl_std::__invoke_acos<half4>(x);
}
inline half8 acos(half8 x) __NOEXC {
  return __sycl_std::__invoke_acos<half8>(x);
}
inline half16 acos(half16 x) __NOEXC {
  return __sycl_std::__invoke_acos<half16>(x);
}

// genfloat acosh (genfloat x)
inline float acosh(float x) __NOEXC {
  return __sycl_std::__invoke_acosh<float>(x);
}
inline float2 acosh(float2 x) __NOEXC {
  return __sycl_std::__invoke_acosh<float2>(x);
}
inline float3 acosh(float3 x) __NOEXC {
  return __sycl_std::__invoke_acosh<float3>(x);
}
inline float4 acosh(float4 x) __NOEXC {
  return __sycl_std::__invoke_acosh<float4>(x);
}
inline float8 acosh(float8 x) __NOEXC {
  return __sycl_std::__invoke_acosh<float8>(x);
}
inline float16 acosh(float16 x) __NOEXC {
  return __sycl_std::__invoke_acosh<float16>(x);
}
inline double acosh(double x) __NOEXC {
  return __sycl_std::__invoke_acosh<double>(x);
}
inline double2 acosh(double2 x) __NOEXC {
  return __sycl_std::__invoke_acosh<double2>(x);
}
inline double3 acosh(double3 x) __NOEXC {
  return __sycl_std::__invoke_acosh<double3>(x);
}
inline double4 acosh(double4 x) __NOEXC {
  return __sycl_std::__invoke_acosh<double4>(x);
}
inline double8 acosh(double8 x) __NOEXC {
  return __sycl_std::__invoke_acosh<double8>(x);
}
inline double16 acosh(double16 x) __NOEXC {
  return __sycl_std::__invoke_acosh<double16>(x);
}
inline half acosh(half x) __NOEXC {
  return __sycl_std::__invoke_acosh<half>(x);
}
inline half2 acosh(half2 x) __NOEXC {
  return __sycl_std::__invoke_acosh<half2>(x);
}
inline half3 acosh(half3 x) __NOEXC {
  return __sycl_std::__invoke_acosh<half3>(x);
}
inline half4 acosh(half4 x) __NOEXC {
  return __sycl_std::__invoke_acosh<half4>(x);
}
inline half8 acosh(half8 x) __NOEXC {
  return __sycl_std::__invoke_acosh<half8>(x);
}
inline half16 acosh(half16 x) __NOEXC {
  return __sycl_std::__invoke_acosh<half16>(x);
}

// genfloat acospi (genfloat x)
inline float acospi(float x) __NOEXC {
  return __sycl_std::__invoke_acospi<float>(x);
}
inline float2 acospi(float2 x) __NOEXC {
  return __sycl_std::__invoke_acospi<float2>(x);
}
inline float3 acospi(float3 x) __NOEXC {
  return __sycl_std::__invoke_acospi<float3>(x);
}
inline float4 acospi(float4 x) __NOEXC {
  return __sycl_std::__invoke_acospi<float4>(x);
}
inline float8 acospi(float8 x) __NOEXC {
  return __sycl_std::__invoke_acospi<float8>(x);
}
inline float16 acospi(float16 x) __NOEXC {
  return __sycl_std::__invoke_acospi<float16>(x);
}
inline double acospi(double x) __NOEXC {
  return __sycl_std::__invoke_acospi<double>(x);
}
inline double2 acospi(double2 x) __NOEXC {
  return __sycl_std::__invoke_acospi<double2>(x);
}
inline double3 acospi(double3 x) __NOEXC {
  return __sycl_std::__invoke_acospi<double3>(x);
}
inline double4 acospi(double4 x) __NOEXC {
  return __sycl_std::__invoke_acospi<double4>(x);
}
inline double8 acospi(double8 x) __NOEXC {
  return __sycl_std::__invoke_acospi<double8>(x);
}
inline double16 acospi(double16 x) __NOEXC {
  return __sycl_std::__invoke_acospi<double16>(x);
}
inline half acospi(half x) __NOEXC {
  return __sycl_std::__invoke_acospi<half>(x);
}
inline half2 acospi(half2 x) __NOEXC {
  return __sycl_std::__invoke_acospi<half2>(x);
}
inline half3 acospi(half3 x) __NOEXC {
  return __sycl_std::__invoke_acospi<half3>(x);
}
inline half4 acospi(half4 x) __NOEXC {
  return __sycl_std::__invoke_acospi<half4>(x);
}
inline half8 acospi(half8 x) __NOEXC {
  return __sycl_std::__invoke_acospi<half8>(x);
}
inline half16 acospi(half16 x) __NOEXC {
  return __sycl_std::__invoke_acospi<half16>(x);
}

// genfloat asin (genfloat x)
inline float asin(float x) __NOEXC {
  return __sycl_std::__invoke_asin<float>(x);
}
inline float2 asin(float2 x) __NOEXC {
  return __sycl_std::__invoke_asin<float2>(x);
}
inline float3 asin(float3 x) __NOEXC {
  return __sycl_std::__invoke_asin<float3>(x);
}
inline float4 asin(float4 x) __NOEXC {
  return __sycl_std::__invoke_asin<float4>(x);
}
inline float8 asin(float8 x) __NOEXC {
  return __sycl_std::__invoke_asin<float8>(x);
}
inline float16 asin(float16 x) __NOEXC {
  return __sycl_std::__invoke_asin<float16>(x);
}
inline double asin(double x) __NOEXC {
  return __sycl_std::__invoke_asin<double>(x);
}
inline double2 asin(double2 x) __NOEXC {
  return __sycl_std::__invoke_asin<double2>(x);
}
inline double3 asin(double3 x) __NOEXC {
  return __sycl_std::__invoke_asin<double3>(x);
}
inline double4 asin(double4 x) __NOEXC {
  return __sycl_std::__invoke_asin<double4>(x);
}
inline double8 asin(double8 x) __NOEXC {
  return __sycl_std::__invoke_asin<double8>(x);
}
inline double16 asin(double16 x) __NOEXC {
  return __sycl_std::__invoke_asin<double16>(x);
}
inline half asin(half x) __NOEXC { return __sycl_std::__invoke_asin<half>(x); }
inline half2 asin(half2 x) __NOEXC {
  return __sycl_std::__invoke_asin<half2>(x);
}
inline half3 asin(half3 x) __NOEXC {
  return __sycl_std::__invoke_asin<half3>(x);
}
inline half4 asin(half4 x) __NOEXC {
  return __sycl_std::__invoke_asin<half4>(x);
}
inline half8 asin(half8 x) __NOEXC {
  return __sycl_std::__invoke_asin<half8>(x);
}
inline half16 asin(half16 x) __NOEXC {
  return __sycl_std::__invoke_asin<half16>(x);
}

// genfloat asinh (genfloat x)
inline float asinh(float x) __NOEXC {
  return __sycl_std::__invoke_asinh<float>(x);
}
inline float2 asinh(float2 x) __NOEXC {
  return __sycl_std::__invoke_asinh<float2>(x);
}
inline float3 asinh(float3 x) __NOEXC {
  return __sycl_std::__invoke_asinh<float3>(x);
}
inline float4 asinh(float4 x) __NOEXC {
  return __sycl_std::__invoke_asinh<float4>(x);
}
inline float8 asinh(float8 x) __NOEXC {
  return __sycl_std::__invoke_asinh<float8>(x);
}
inline float16 asinh(float16 x) __NOEXC {
  return __sycl_std::__invoke_asinh<float16>(x);
}
inline double asinh(double x) __NOEXC {
  return __sycl_std::__invoke_asinh<double>(x);
}
inline double2 asinh(double2 x) __NOEXC {
  return __sycl_std::__invoke_asinh<double2>(x);
}
inline double3 asinh(double3 x) __NOEXC {
  return __sycl_std::__invoke_asinh<double3>(x);
}
inline double4 asinh(double4 x) __NOEXC {
  return __sycl_std::__invoke_asinh<double4>(x);
}
inline double8 asinh(double8 x) __NOEXC {
  return __sycl_std::__invoke_asinh<double8>(x);
}
inline double16 asinh(double16 x) __NOEXC {
  return __sycl_std::__invoke_asinh<double16>(x);
}
inline half asinh(half x) __NOEXC {
  return __sycl_std::__invoke_asinh<half>(x);
}
inline half2 asinh(half2 x) __NOEXC {
  return __sycl_std::__invoke_asinh<half2>(x);
}
inline half3 asinh(half3 x) __NOEXC {
  return __sycl_std::__invoke_asinh<half3>(x);
}
inline half4 asinh(half4 x) __NOEXC {
  return __sycl_std::__invoke_asinh<half4>(x);
}
inline half8 asinh(half8 x) __NOEXC {
  return __sycl_std::__invoke_asinh<half8>(x);
}
inline half16 asinh(half16 x) __NOEXC {
  return __sycl_std::__invoke_asinh<half16>(x);
}

// genfloat asinpi (genfloat x)
inline float asinpi(float x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float>(x);
}
inline float2 asinpi(float2 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float2>(x);
}
inline float3 asinpi(float3 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float3>(x);
}
inline float4 asinpi(float4 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float4>(x);
}
inline float8 asinpi(float8 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float8>(x);
}
inline float16 asinpi(float16 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<float16>(x);
}
inline double asinpi(double x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double>(x);
}
inline double2 asinpi(double2 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double2>(x);
}
inline double3 asinpi(double3 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double3>(x);
}
inline double4 asinpi(double4 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double4>(x);
}
inline double8 asinpi(double8 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double8>(x);
}
inline double16 asinpi(double16 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<double16>(x);
}
inline half asinpi(half x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half>(x);
}
inline half2 asinpi(half2 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half2>(x);
}
inline half3 asinpi(half3 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half3>(x);
}
inline half4 asinpi(half4 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half4>(x);
}
inline half8 asinpi(half8 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half8>(x);
}
inline half16 asinpi(half16 x) __NOEXC {
  return __sycl_std::__invoke_asinpi<half16>(x);
}

// genfloat atan (genfloat y_over_x)
inline float atan(float y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float>(y_over_x);
}
inline float2 atan(float2 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float2>(y_over_x);
}
inline float3 atan(float3 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float3>(y_over_x);
}
inline float4 atan(float4 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float4>(y_over_x);
}
inline float8 atan(float8 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float8>(y_over_x);
}
inline float16 atan(float16 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<float16>(y_over_x);
}
inline double atan(double y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double>(y_over_x);
}
inline double2 atan(double2 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double2>(y_over_x);
}
inline double3 atan(double3 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double3>(y_over_x);
}
inline double4 atan(double4 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double4>(y_over_x);
}
inline double8 atan(double8 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double8>(y_over_x);
}
inline double16 atan(double16 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<double16>(y_over_x);
}
inline half atan(half y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half>(y_over_x);
}
inline half2 atan(half2 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half2>(y_over_x);
}
inline half3 atan(half3 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half3>(y_over_x);
}
inline half4 atan(half4 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half4>(y_over_x);
}
inline half8 atan(half8 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half8>(y_over_x);
}
inline half16 atan(half16 y_over_x) __NOEXC {
  return __sycl_std::__invoke_atan<half16>(y_over_x);
}

// genfloat atan2 (genfloat y, genfloat x)
inline float atan2(float y, float x) __NOEXC {
  return __sycl_std::__invoke_atan2<float>(y, x);
}
inline float2 atan2(float2 y, float2 x) __NOEXC {
  return __sycl_std::__invoke_atan2<float2>(y, x);
}
inline float3 atan2(float3 y, float3 x) __NOEXC {
  return __sycl_std::__invoke_atan2<float3>(y, x);
}
inline float4 atan2(float4 y, float4 x) __NOEXC {
  return __sycl_std::__invoke_atan2<float4>(y, x);
}
inline float8 atan2(float8 y, float8 x) __NOEXC {
  return __sycl_std::__invoke_atan2<float8>(y, x);
}
inline float16 atan2(float16 y, float16 x) __NOEXC {
  return __sycl_std::__invoke_atan2<float16>(y, x);
}
inline double atan2(double y, double x) __NOEXC {
  return __sycl_std::__invoke_atan2<double>(y, x);
}
inline double2 atan2(double2 y, double2 x) __NOEXC {
  return __sycl_std::__invoke_atan2<double2>(y, x);
}
inline double3 atan2(double3 y, double3 x) __NOEXC {
  return __sycl_std::__invoke_atan2<double3>(y, x);
}
inline double4 atan2(double4 y, double4 x) __NOEXC {
  return __sycl_std::__invoke_atan2<double4>(y, x);
}
inline double8 atan2(double8 y, double8 x) __NOEXC {
  return __sycl_std::__invoke_atan2<double8>(y, x);
}
inline double16 atan2(double16 y, double16 x) __NOEXC {
  return __sycl_std::__invoke_atan2<double16>(y, x);
}
inline half atan2(half y, half x) __NOEXC {
  return __sycl_std::__invoke_atan2<half>(y, x);
}
inline half2 atan2(half2 y, half2 x) __NOEXC {
  return __sycl_std::__invoke_atan2<half2>(y, x);
}
inline half3 atan2(half3 y, half3 x) __NOEXC {
  return __sycl_std::__invoke_atan2<half3>(y, x);
}
inline half4 atan2(half4 y, half4 x) __NOEXC {
  return __sycl_std::__invoke_atan2<half4>(y, x);
}
inline half8 atan2(half8 y, half8 x) __NOEXC {
  return __sycl_std::__invoke_atan2<half8>(y, x);
}
inline half16 atan2(half16 y, half16 x) __NOEXC {
  return __sycl_std::__invoke_atan2<half16>(y, x);
}

// genfloat atanh (genfloat x)
inline float atanh(float x) __NOEXC {
  return __sycl_std::__invoke_atanh<float>(x);
}
inline float2 atanh(float2 x) __NOEXC {
  return __sycl_std::__invoke_atanh<float2>(x);
}
inline float3 atanh(float3 x) __NOEXC {
  return __sycl_std::__invoke_atanh<float3>(x);
}
inline float4 atanh(float4 x) __NOEXC {
  return __sycl_std::__invoke_atanh<float4>(x);
}
inline float8 atanh(float8 x) __NOEXC {
  return __sycl_std::__invoke_atanh<float8>(x);
}
inline float16 atanh(float16 x) __NOEXC {
  return __sycl_std::__invoke_atanh<float16>(x);
}
inline double atanh(double x) __NOEXC {
  return __sycl_std::__invoke_atanh<double>(x);
}
inline double2 atanh(double2 x) __NOEXC {
  return __sycl_std::__invoke_atanh<double2>(x);
}
inline double3 atanh(double3 x) __NOEXC {
  return __sycl_std::__invoke_atanh<double3>(x);
}
inline double4 atanh(double4 x) __NOEXC {
  return __sycl_std::__invoke_atanh<double4>(x);
}
inline double8 atanh(double8 x) __NOEXC {
  return __sycl_std::__invoke_atanh<double8>(x);
}
inline double16 atanh(double16 x) __NOEXC {
  return __sycl_std::__invoke_atanh<double16>(x);
}
inline half atanh(half x) __NOEXC {
  return __sycl_std::__invoke_atanh<half>(x);
}
inline half2 atanh(half2 x) __NOEXC {
  return __sycl_std::__invoke_atanh<half2>(x);
}
inline half3 atanh(half3 x) __NOEXC {
  return __sycl_std::__invoke_atanh<half3>(x);
}
inline half4 atanh(half4 x) __NOEXC {
  return __sycl_std::__invoke_atanh<half4>(x);
}
inline half8 atanh(half8 x) __NOEXC {
  return __sycl_std::__invoke_atanh<half8>(x);
}
inline half16 atanh(half16 x) __NOEXC {
  return __sycl_std::__invoke_atanh<half16>(x);
}

// genfloat atanpi (genfloat x)
inline float atanpi(float x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float>(x);
}
inline float2 atanpi(float2 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float2>(x);
}
inline float3 atanpi(float3 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float3>(x);
}
inline float4 atanpi(float4 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float4>(x);
}
inline float8 atanpi(float8 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float8>(x);
}
inline float16 atanpi(float16 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<float16>(x);
}
inline double atanpi(double x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double>(x);
}
inline double2 atanpi(double2 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double2>(x);
}
inline double3 atanpi(double3 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double3>(x);
}
inline double4 atanpi(double4 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double4>(x);
}
inline double8 atanpi(double8 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double8>(x);
}
inline double16 atanpi(double16 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<double16>(x);
}
inline half atanpi(half x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half>(x);
}
inline half2 atanpi(half2 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half2>(x);
}
inline half3 atanpi(half3 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half3>(x);
}
inline half4 atanpi(half4 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half4>(x);
}
inline half8 atanpi(half8 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half8>(x);
}
inline half16 atanpi(half16 x) __NOEXC {
  return __sycl_std::__invoke_atanpi<half16>(x);
}

// genfloat atan2pi (genfloat y, genfloat x)
inline float atan2pi(float y, float x) {
  return __sycl_std::__invoke_atan2pi<float>(y, x);
}
inline float2 atan2pi(float2 y, float2 x) {
  return __sycl_std::__invoke_atan2pi<float2>(y, x);
}
inline float3 atan2pi(float3 y, float3 x) {
  return __sycl_std::__invoke_atan2pi<float3>(y, x);
}
inline float4 atan2pi(float4 y, float4 x) {
  return __sycl_std::__invoke_atan2pi<float4>(y, x);
}
inline float8 atan2pi(float8 y, float8 x) {
  return __sycl_std::__invoke_atan2pi<float8>(y, x);
}
inline float16 atan2pi(float16 y, float16 x) {
  return __sycl_std::__invoke_atan2pi<float16>(y, x);
}
inline double atan2pi(double y, double x) {
  return __sycl_std::__invoke_atan2pi<double>(y, x);
}
inline double2 atan2pi(double2 y, double2 x) {
  return __sycl_std::__invoke_atan2pi<double2>(y, x);
}
inline double3 atan2pi(double3 y, double3 x) {
  return __sycl_std::__invoke_atan2pi<double3>(y, x);
}
inline double4 atan2pi(double4 y, double4 x) {
  return __sycl_std::__invoke_atan2pi<double4>(y, x);
}
inline double8 atan2pi(double8 y, double8 x) {
  return __sycl_std::__invoke_atan2pi<double8>(y, x);
}
inline double16 atan2pi(double16 y, double16 x) {
  return __sycl_std::__invoke_atan2pi<double16>(y, x);
}
inline half atan2pi(half y, half x) {
  return __sycl_std::__invoke_atan2pi<half>(y, x);
}
inline half2 atan2pi(half2 y, half2 x) {
  return __sycl_std::__invoke_atan2pi<half2>(y, x);
}
inline half3 atan2pi(half3 y, half3 x) {
  return __sycl_std::__invoke_atan2pi<half3>(y, x);
}
inline half4 atan2pi(half4 y, half4 x) {
  return __sycl_std::__invoke_atan2pi<half4>(y, x);
}
inline half8 atan2pi(half8 y, half8 x) {
  return __sycl_std::__invoke_atan2pi<half8>(y, x);
}
inline half16 atan2pi(half16 y, half16 x) {
  return __sycl_std::__invoke_atan2pi<half16>(y, x);
}

// genfloat cbrt (genfloat x)
inline float cbrt(float x) { return __sycl_std::__invoke_cbrt<float>(x); }
inline float2 cbrt(float2 x) { return __sycl_std::__invoke_cbrt<float2>(x); }
inline float3 cbrt(float3 x) { return __sycl_std::__invoke_cbrt<float3>(x); }
inline float4 cbrt(float4 x) { return __sycl_std::__invoke_cbrt<float4>(x); }
inline float8 cbrt(float8 x) { return __sycl_std::__invoke_cbrt<float8>(x); }
inline float16 cbrt(float16 x) { return __sycl_std::__invoke_cbrt<float16>(x); }
inline double cbrt(double x) { return __sycl_std::__invoke_cbrt<double>(x); }
inline double2 cbrt(double2 x) { return __sycl_std::__invoke_cbrt<double2>(x); }
inline double3 cbrt(double3 x) { return __sycl_std::__invoke_cbrt<double3>(x); }
inline double4 cbrt(double4 x) { return __sycl_std::__invoke_cbrt<double4>(x); }
inline double8 cbrt(double8 x) { return __sycl_std::__invoke_cbrt<double8>(x); }
inline double16 cbrt(double16 x) {
  return __sycl_std::__invoke_cbrt<double16>(x);
}
inline half cbrt(half x) { return __sycl_std::__invoke_cbrt<half>(x); }
inline half2 cbrt(half2 x) { return __sycl_std::__invoke_cbrt<half2>(x); }
inline half3 cbrt(half3 x) { return __sycl_std::__invoke_cbrt<half3>(x); }
inline half4 cbrt(half4 x) { return __sycl_std::__invoke_cbrt<half4>(x); }
inline half8 cbrt(half8 x) { return __sycl_std::__invoke_cbrt<half8>(x); }
inline half16 cbrt(half16 x) { return __sycl_std::__invoke_cbrt<half16>(x); }

// genfloat ceil (genfloat x)
inline float ceil(float x) { return __sycl_std::__invoke_ceil<float>(x); }
inline float2 ceil(float2 x) { return __sycl_std::__invoke_ceil<float2>(x); }
inline float3 ceil(float3 x) { return __sycl_std::__invoke_ceil<float3>(x); }
inline float4 ceil(float4 x) { return __sycl_std::__invoke_ceil<float4>(x); }
inline float8 ceil(float8 x) { return __sycl_std::__invoke_ceil<float8>(x); }
inline float16 ceil(float16 x) { return __sycl_std::__invoke_ceil<float16>(x); }
inline double ceil(double x) { return __sycl_std::__invoke_ceil<double>(x); }
inline double2 ceil(double2 x) { return __sycl_std::__invoke_ceil<double2>(x); }
inline double3 ceil(double3 x) { return __sycl_std::__invoke_ceil<double3>(x); }
inline double4 ceil(double4 x) { return __sycl_std::__invoke_ceil<double4>(x); }
inline double8 ceil(double8 x) { return __sycl_std::__invoke_ceil<double8>(x); }
inline double16 ceil(double16 x) {
  return __sycl_std::__invoke_ceil<double16>(x);
}
inline half ceil(half x) { return __sycl_std::__invoke_ceil<half>(x); }
inline half2 ceil(half2 x) { return __sycl_std::__invoke_ceil<half2>(x); }
inline half3 ceil(half3 x) { return __sycl_std::__invoke_ceil<half3>(x); }
inline half4 ceil(half4 x) { return __sycl_std::__invoke_ceil<half4>(x); }
inline half8 ceil(half8 x) { return __sycl_std::__invoke_ceil<half8>(x); }
inline half16 ceil(half16 x) { return __sycl_std::__invoke_ceil<half16>(x); }

// genfloat copysign (genfloat x, genfloat y)
inline float copysign(float x, float y) __NOEXC {
  return __sycl_std::__invoke_copysign<float>(x, y);
}
inline float2 copysign(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_copysign<float2>(x, y);
}
inline float3 copysign(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_copysign<float3>(x, y);
}
inline float4 copysign(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_copysign<float4>(x, y);
}
inline float8 copysign(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_copysign<float8>(x, y);
}
inline float16 copysign(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_copysign<float16>(x, y);
}
inline double copysign(double x, double y) __NOEXC {
  return __sycl_std::__invoke_copysign<double>(x, y);
}
inline double2 copysign(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_copysign<double2>(x, y);
}
inline double3 copysign(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_copysign<double3>(x, y);
}
inline double4 copysign(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_copysign<double4>(x, y);
}
inline double8 copysign(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_copysign<double8>(x, y);
}
inline double16 copysign(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_copysign<double16>(x, y);
}
inline half copysign(half x, half y) __NOEXC {
  return __sycl_std::__invoke_copysign<half>(x, y);
}
inline half2 copysign(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_copysign<half2>(x, y);
}
inline half3 copysign(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_copysign<half3>(x, y);
}
inline half4 copysign(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_copysign<half4>(x, y);
}
inline half8 copysign(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_copysign<half8>(x, y);
}
inline half16 copysign(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_copysign<half16>(x, y);
}

// genfloat cos (genfloat x)
inline float cos(float x) __NOEXC { return __sycl_std::__invoke_cos<float>(x); }
inline float2 cos(float2 x) __NOEXC {
  return __sycl_std::__invoke_cos<float2>(x);
}
inline float3 cos(float3 x) __NOEXC {
  return __sycl_std::__invoke_cos<float3>(x);
}
inline float4 cos(float4 x) __NOEXC {
  return __sycl_std::__invoke_cos<float4>(x);
}
inline float8 cos(float8 x) __NOEXC {
  return __sycl_std::__invoke_cos<float8>(x);
}
inline float16 cos(float16 x) __NOEXC {
  return __sycl_std::__invoke_cos<float16>(x);
}
inline double cos(double x) __NOEXC {
  return __sycl_std::__invoke_cos<double>(x);
}
inline double2 cos(double2 x) __NOEXC {
  return __sycl_std::__invoke_cos<double2>(x);
}
inline double3 cos(double3 x) __NOEXC {
  return __sycl_std::__invoke_cos<double3>(x);
}
inline double4 cos(double4 x) __NOEXC {
  return __sycl_std::__invoke_cos<double4>(x);
}
inline double8 cos(double8 x) __NOEXC {
  return __sycl_std::__invoke_cos<double8>(x);
}
inline double16 cos(double16 x) __NOEXC {
  return __sycl_std::__invoke_cos<double16>(x);
}
inline half cos(half x) __NOEXC { return __sycl_std::__invoke_cos<half>(x); }
inline half2 cos(half2 x) __NOEXC { return __sycl_std::__invoke_cos<half2>(x); }
inline half3 cos(half3 x) __NOEXC { return __sycl_std::__invoke_cos<half3>(x); }
inline half4 cos(half4 x) __NOEXC { return __sycl_std::__invoke_cos<half4>(x); }
inline half8 cos(half8 x) __NOEXC { return __sycl_std::__invoke_cos<half8>(x); }
inline half16 cos(half16 x) __NOEXC {
  return __sycl_std::__invoke_cos<half16>(x);
}

// genfloat cosh (genfloat x)
inline float cosh(float x) __NOEXC {
  return __sycl_std::__invoke_cosh<float>(x);
}
inline float2 cosh(float2 x) __NOEXC {
  return __sycl_std::__invoke_cosh<float2>(x);
}
inline float3 cosh(float3 x) __NOEXC {
  return __sycl_std::__invoke_cosh<float3>(x);
}
inline float4 cosh(float4 x) __NOEXC {
  return __sycl_std::__invoke_cosh<float4>(x);
}
inline float8 cosh(float8 x) __NOEXC {
  return __sycl_std::__invoke_cosh<float8>(x);
}
inline float16 cosh(float16 x) __NOEXC {
  return __sycl_std::__invoke_cosh<float16>(x);
}
inline double cosh(double x) __NOEXC {
  return __sycl_std::__invoke_cosh<double>(x);
}
inline double2 cosh(double2 x) __NOEXC {
  return __sycl_std::__invoke_cosh<double2>(x);
}
inline double3 cosh(double3 x) __NOEXC {
  return __sycl_std::__invoke_cosh<double3>(x);
}
inline double4 cosh(double4 x) __NOEXC {
  return __sycl_std::__invoke_cosh<double4>(x);
}
inline double8 cosh(double8 x) __NOEXC {
  return __sycl_std::__invoke_cosh<double8>(x);
}
inline double16 cosh(double16 x) __NOEXC {
  return __sycl_std::__invoke_cosh<double16>(x);
}
inline half cosh(half x) __NOEXC { return __sycl_std::__invoke_cosh<half>(x); }
inline half2 cosh(half2 x) __NOEXC {
  return __sycl_std::__invoke_cosh<half2>(x);
}
inline half3 cosh(half3 x) __NOEXC {
  return __sycl_std::__invoke_cosh<half3>(x);
}
inline half4 cosh(half4 x) __NOEXC {
  return __sycl_std::__invoke_cosh<half4>(x);
}
inline half8 cosh(half8 x) __NOEXC {
  return __sycl_std::__invoke_cosh<half8>(x);
}
inline half16 cosh(half16 x) __NOEXC {
  return __sycl_std::__invoke_cosh<half16>(x);
}

// genfloat cospi (genfloat x)
inline float cospi(float x) __NOEXC {
  return __sycl_std::__invoke_cospi<float>(x);
}
inline float2 cospi(float2 x) __NOEXC {
  return __sycl_std::__invoke_cospi<float2>(x);
}
inline float3 cospi(float3 x) __NOEXC {
  return __sycl_std::__invoke_cospi<float3>(x);
}
inline float4 cospi(float4 x) __NOEXC {
  return __sycl_std::__invoke_cospi<float4>(x);
}
inline float8 cospi(float8 x) __NOEXC {
  return __sycl_std::__invoke_cospi<float8>(x);
}
inline float16 cospi(float16 x) __NOEXC {
  return __sycl_std::__invoke_cospi<float16>(x);
}
inline double cospi(double x) __NOEXC {
  return __sycl_std::__invoke_cospi<double>(x);
}
inline double2 cospi(double2 x) __NOEXC {
  return __sycl_std::__invoke_cospi<double2>(x);
}
inline double3 cospi(double3 x) __NOEXC {
  return __sycl_std::__invoke_cospi<double3>(x);
}
inline double4 cospi(double4 x) __NOEXC {
  return __sycl_std::__invoke_cospi<double4>(x);
}
inline double8 cospi(double8 x) __NOEXC {
  return __sycl_std::__invoke_cospi<double8>(x);
}
inline double16 cospi(double16 x) __NOEXC {
  return __sycl_std::__invoke_cospi<double16>(x);
}
inline half cospi(half x) __NOEXC {
  return __sycl_std::__invoke_cospi<half>(x);
}
inline half2 cospi(half2 x) __NOEXC {
  return __sycl_std::__invoke_cospi<half2>(x);
}
inline half3 cospi(half3 x) __NOEXC {
  return __sycl_std::__invoke_cospi<half3>(x);
}
inline half4 cospi(half4 x) __NOEXC {
  return __sycl_std::__invoke_cospi<half4>(x);
}
inline half8 cospi(half8 x) __NOEXC {
  return __sycl_std::__invoke_cospi<half8>(x);
}
inline half16 cospi(half16 x) __NOEXC {
  return __sycl_std::__invoke_cospi<half16>(x);
}

// genfloat erfc (genfloat x)
inline float erfc(float x) __NOEXC {
  return __sycl_std::__invoke_erfc<float>(x);
}
inline float2 erfc(float2 x) __NOEXC {
  return __sycl_std::__invoke_erfc<float2>(x);
}
inline float3 erfc(float3 x) __NOEXC {
  return __sycl_std::__invoke_erfc<float3>(x);
}
inline float4 erfc(float4 x) __NOEXC {
  return __sycl_std::__invoke_erfc<float4>(x);
}
inline float8 erfc(float8 x) __NOEXC {
  return __sycl_std::__invoke_erfc<float8>(x);
}
inline float16 erfc(float16 x) __NOEXC {
  return __sycl_std::__invoke_erfc<float16>(x);
}
inline double erfc(double x) __NOEXC {
  return __sycl_std::__invoke_erfc<double>(x);
}
inline double2 erfc(double2 x) __NOEXC {
  return __sycl_std::__invoke_erfc<double2>(x);
}
inline double3 erfc(double3 x) __NOEXC {
  return __sycl_std::__invoke_erfc<double3>(x);
}
inline double4 erfc(double4 x) __NOEXC {
  return __sycl_std::__invoke_erfc<double4>(x);
}
inline double8 erfc(double8 x) __NOEXC {
  return __sycl_std::__invoke_erfc<double8>(x);
}
inline double16 erfc(double16 x) __NOEXC {
  return __sycl_std::__invoke_erfc<double16>(x);
}
inline half erfc(half x) __NOEXC { return __sycl_std::__invoke_erfc<half>(x); }
inline half2 erfc(half2 x) __NOEXC {
  return __sycl_std::__invoke_erfc<half2>(x);
}
inline half3 erfc(half3 x) __NOEXC {
  return __sycl_std::__invoke_erfc<half3>(x);
}
inline half4 erfc(half4 x) __NOEXC {
  return __sycl_std::__invoke_erfc<half4>(x);
}
inline half8 erfc(half8 x) __NOEXC {
  return __sycl_std::__invoke_erfc<half8>(x);
}
inline half16 erfc(half16 x) __NOEXC {
  return __sycl_std::__invoke_erfc<half16>(x);
}

// genfloat erf (genfloat x)
inline float erf(float x) __NOEXC { return __sycl_std::__invoke_erf<float>(x); }
inline float2 erf(float2 x) __NOEXC {
  return __sycl_std::__invoke_erf<float2>(x);
}
inline float3 erf(float3 x) __NOEXC {
  return __sycl_std::__invoke_erf<float3>(x);
}
inline float4 erf(float4 x) __NOEXC {
  return __sycl_std::__invoke_erf<float4>(x);
}
inline float8 erf(float8 x) __NOEXC {
  return __sycl_std::__invoke_erf<float8>(x);
}
inline float16 erf(float16 x) __NOEXC {
  return __sycl_std::__invoke_erf<float16>(x);
}
inline double erf(double x) __NOEXC {
  return __sycl_std::__invoke_erf<double>(x);
}
inline double2 erf(double2 x) __NOEXC {
  return __sycl_std::__invoke_erf<double2>(x);
}
inline double3 erf(double3 x) __NOEXC {
  return __sycl_std::__invoke_erf<double3>(x);
}
inline double4 erf(double4 x) __NOEXC {
  return __sycl_std::__invoke_erf<double4>(x);
}
inline double8 erf(double8 x) __NOEXC {
  return __sycl_std::__invoke_erf<double8>(x);
}
inline double16 erf(double16 x) __NOEXC {
  return __sycl_std::__invoke_erf<double16>(x);
}
inline half erf(half x) __NOEXC { return __sycl_std::__invoke_erf<half>(x); }
inline half2 erf(half2 x) __NOEXC { return __sycl_std::__invoke_erf<half2>(x); }
inline half3 erf(half3 x) __NOEXC { return __sycl_std::__invoke_erf<half3>(x); }
inline half4 erf(half4 x) __NOEXC { return __sycl_std::__invoke_erf<half4>(x); }
inline half8 erf(half8 x) __NOEXC { return __sycl_std::__invoke_erf<half8>(x); }
inline half16 erf(half16 x) __NOEXC {
  return __sycl_std::__invoke_erf<half16>(x);
}

// genfloat exp (genfloat x )
inline float exp(float x) __NOEXC { return __sycl_std::__invoke_exp<float>(x); }
inline float2 exp(float2 x) __NOEXC {
  return __sycl_std::__invoke_exp<float2>(x);
}
inline float3 exp(float3 x) __NOEXC {
  return __sycl_std::__invoke_exp<float3>(x);
}
inline float4 exp(float4 x) __NOEXC {
  return __sycl_std::__invoke_exp<float4>(x);
}
inline float8 exp(float8 x) __NOEXC {
  return __sycl_std::__invoke_exp<float8>(x);
}
inline float16 exp(float16 x) __NOEXC {
  return __sycl_std::__invoke_exp<float16>(x);
}
inline double exp(double x) __NOEXC {
  return __sycl_std::__invoke_exp<double>(x);
}
inline double2 exp(double2 x) __NOEXC {
  return __sycl_std::__invoke_exp<double2>(x);
}
inline double3 exp(double3 x) __NOEXC {
  return __sycl_std::__invoke_exp<double3>(x);
}
inline double4 exp(double4 x) __NOEXC {
  return __sycl_std::__invoke_exp<double4>(x);
}
inline double8 exp(double8 x) __NOEXC {
  return __sycl_std::__invoke_exp<double8>(x);
}
inline double16 exp(double16 x) __NOEXC {
  return __sycl_std::__invoke_exp<double16>(x);
}
inline half exp(half x) __NOEXC { return __sycl_std::__invoke_exp<half>(x); }
inline half2 exp(half2 x) __NOEXC { return __sycl_std::__invoke_exp<half2>(x); }
inline half3 exp(half3 x) __NOEXC { return __sycl_std::__invoke_exp<half3>(x); }
inline half4 exp(half4 x) __NOEXC { return __sycl_std::__invoke_exp<half4>(x); }
inline half8 exp(half8 x) __NOEXC { return __sycl_std::__invoke_exp<half8>(x); }
inline half16 exp(half16 x) __NOEXC {
  return __sycl_std::__invoke_exp<half16>(x);
}

// genfloat exp2 (genfloat x)
inline float exp2(float x) __NOEXC {
  return __sycl_std::__invoke_exp2<float>(x);
}
inline float2 exp2(float2 x) __NOEXC {
  return __sycl_std::__invoke_exp2<float2>(x);
}
inline float3 exp2(float3 x) __NOEXC {
  return __sycl_std::__invoke_exp2<float3>(x);
}
inline float4 exp2(float4 x) __NOEXC {
  return __sycl_std::__invoke_exp2<float4>(x);
}
inline float8 exp2(float8 x) __NOEXC {
  return __sycl_std::__invoke_exp2<float8>(x);
}
inline float16 exp2(float16 x) __NOEXC {
  return __sycl_std::__invoke_exp2<float16>(x);
}
inline double exp2(double x) __NOEXC {
  return __sycl_std::__invoke_exp2<double>(x);
}
inline double2 exp2(double2 x) __NOEXC {
  return __sycl_std::__invoke_exp2<double2>(x);
}
inline double3 exp2(double3 x) __NOEXC {
  return __sycl_std::__invoke_exp2<double3>(x);
}
inline double4 exp2(double4 x) __NOEXC {
  return __sycl_std::__invoke_exp2<double4>(x);
}
inline double8 exp2(double8 x) __NOEXC {
  return __sycl_std::__invoke_exp2<double8>(x);
}
inline double16 exp2(double16 x) __NOEXC {
  return __sycl_std::__invoke_exp2<double16>(x);
}
inline half exp2(half x) __NOEXC { return __sycl_std::__invoke_exp2<half>(x); }
inline half2 exp2(half2 x) __NOEXC {
  return __sycl_std::__invoke_exp2<half2>(x);
}
inline half3 exp2(half3 x) __NOEXC {
  return __sycl_std::__invoke_exp2<half3>(x);
}
inline half4 exp2(half4 x) __NOEXC {
  return __sycl_std::__invoke_exp2<half4>(x);
}
inline half8 exp2(half8 x) __NOEXC {
  return __sycl_std::__invoke_exp2<half8>(x);
}
inline half16 exp2(half16 x) __NOEXC {
  return __sycl_std::__invoke_exp2<half16>(x);
}

// genfloat exp10 (genfloat x)
inline float exp10(float x) __NOEXC {
  return __sycl_std::__invoke_exp10<float>(x);
}
inline float2 exp10(float2 x) __NOEXC {
  return __sycl_std::__invoke_exp10<float2>(x);
}
inline float3 exp10(float3 x) __NOEXC {
  return __sycl_std::__invoke_exp10<float3>(x);
}
inline float4 exp10(float4 x) __NOEXC {
  return __sycl_std::__invoke_exp10<float4>(x);
}
inline float8 exp10(float8 x) __NOEXC {
  return __sycl_std::__invoke_exp10<float8>(x);
}
inline float16 exp10(float16 x) __NOEXC {
  return __sycl_std::__invoke_exp10<float16>(x);
}
inline double exp10(double x) __NOEXC {
  return __sycl_std::__invoke_exp10<double>(x);
}
inline double2 exp10(double2 x) __NOEXC {
  return __sycl_std::__invoke_exp10<double2>(x);
}
inline double3 exp10(double3 x) __NOEXC {
  return __sycl_std::__invoke_exp10<double3>(x);
}
inline double4 exp10(double4 x) __NOEXC {
  return __sycl_std::__invoke_exp10<double4>(x);
}
inline double8 exp10(double8 x) __NOEXC {
  return __sycl_std::__invoke_exp10<double8>(x);
}
inline double16 exp10(double16 x) __NOEXC {
  return __sycl_std::__invoke_exp10<double16>(x);
}
inline half exp10(half x) __NOEXC {
  return __sycl_std::__invoke_exp10<half>(x);
}
inline half2 exp10(half2 x) __NOEXC {
  return __sycl_std::__invoke_exp10<half2>(x);
}
inline half3 exp10(half3 x) __NOEXC {
  return __sycl_std::__invoke_exp10<half3>(x);
}
inline half4 exp10(half4 x) __NOEXC {
  return __sycl_std::__invoke_exp10<half4>(x);
}
inline half8 exp10(half8 x) __NOEXC {
  return __sycl_std::__invoke_exp10<half8>(x);
}
inline half16 exp10(half16 x) __NOEXC {
  return __sycl_std::__invoke_exp10<half16>(x);
}

// genfloat expm1 (genfloat x)
inline float expm1(float x) __NOEXC {
  return __sycl_std::__invoke_expm1<float>(x);
}
inline float2 expm1(float2 x) __NOEXC {
  return __sycl_std::__invoke_expm1<float2>(x);
}
inline float3 expm1(float3 x) __NOEXC {
  return __sycl_std::__invoke_expm1<float3>(x);
}
inline float4 expm1(float4 x) __NOEXC {
  return __sycl_std::__invoke_expm1<float4>(x);
}
inline float8 expm1(float8 x) __NOEXC {
  return __sycl_std::__invoke_expm1<float8>(x);
}
inline float16 expm1(float16 x) __NOEXC {
  return __sycl_std::__invoke_expm1<float16>(x);
}
inline double expm1(double x) __NOEXC {
  return __sycl_std::__invoke_expm1<double>(x);
}
inline double2 expm1(double2 x) __NOEXC {
  return __sycl_std::__invoke_expm1<double2>(x);
}
inline double3 expm1(double3 x) __NOEXC {
  return __sycl_std::__invoke_expm1<double3>(x);
}
inline double4 expm1(double4 x) __NOEXC {
  return __sycl_std::__invoke_expm1<double4>(x);
}
inline double8 expm1(double8 x) __NOEXC {
  return __sycl_std::__invoke_expm1<double8>(x);
}
inline double16 expm1(double16 x) __NOEXC {
  return __sycl_std::__invoke_expm1<double16>(x);
}
inline half expm1(half x) __NOEXC {
  return __sycl_std::__invoke_expm1<half>(x);
}
inline half2 expm1(half2 x) __NOEXC {
  return __sycl_std::__invoke_expm1<half2>(x);
}
inline half3 expm1(half3 x) __NOEXC {
  return __sycl_std::__invoke_expm1<half3>(x);
}
inline half4 expm1(half4 x) __NOEXC {
  return __sycl_std::__invoke_expm1<half4>(x);
}
inline half8 expm1(half8 x) __NOEXC {
  return __sycl_std::__invoke_expm1<half8>(x);
}
inline half16 expm1(half16 x) __NOEXC {
  return __sycl_std::__invoke_expm1<half16>(x);
}

// genfloat fabs (genfloat x)
inline float fabs(float x) __NOEXC {
  return __sycl_std::__invoke_fabs<float>(x);
}
inline float2 fabs(float2 x) __NOEXC {
  return __sycl_std::__invoke_fabs<float2>(x);
}
inline float3 fabs(float3 x) __NOEXC {
  return __sycl_std::__invoke_fabs<float3>(x);
}
inline float4 fabs(float4 x) __NOEXC {
  return __sycl_std::__invoke_fabs<float4>(x);
}
inline float8 fabs(float8 x) __NOEXC {
  return __sycl_std::__invoke_fabs<float8>(x);
}
inline float16 fabs(float16 x) __NOEXC {
  return __sycl_std::__invoke_fabs<float16>(x);
}
inline double fabs(double x) __NOEXC {
  return __sycl_std::__invoke_fabs<double>(x);
}
inline double2 fabs(double2 x) __NOEXC {
  return __sycl_std::__invoke_fabs<double2>(x);
}
inline double3 fabs(double3 x) __NOEXC {
  return __sycl_std::__invoke_fabs<double3>(x);
}
inline double4 fabs(double4 x) __NOEXC {
  return __sycl_std::__invoke_fabs<double4>(x);
}
inline double8 fabs(double8 x) __NOEXC {
  return __sycl_std::__invoke_fabs<double8>(x);
}
inline double16 fabs(double16 x) __NOEXC {
  return __sycl_std::__invoke_fabs<double16>(x);
}
inline half fabs(half x) __NOEXC { return __sycl_std::__invoke_fabs<half>(x); }
inline half2 fabs(half2 x) __NOEXC {
  return __sycl_std::__invoke_fabs<half2>(x);
}
inline half3 fabs(half3 x) __NOEXC {
  return __sycl_std::__invoke_fabs<half3>(x);
}
inline half4 fabs(half4 x) __NOEXC {
  return __sycl_std::__invoke_fabs<half4>(x);
}
inline half8 fabs(half8 x) __NOEXC {
  return __sycl_std::__invoke_fabs<half8>(x);
}
inline half16 fabs(half16 x) __NOEXC {
  return __sycl_std::__invoke_fabs<half16>(x);
}

// genfloat fdim (genfloat x, genfloat y)
inline float fdim(float x, float y) __NOEXC {
  return __sycl_std::__invoke_fdim<float>(x, y);
}
inline float2 fdim(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_fdim<float2>(x, y);
}
inline float3 fdim(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_fdim<float3>(x, y);
}
inline float4 fdim(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_fdim<float4>(x, y);
}
inline float8 fdim(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_fdim<float8>(x, y);
}
inline float16 fdim(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_fdim<float16>(x, y);
}
inline double fdim(double x, double y) __NOEXC {
  return __sycl_std::__invoke_fdim<double>(x, y);
}
inline double2 fdim(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_fdim<double2>(x, y);
}
inline double3 fdim(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_fdim<double3>(x, y);
}
inline double4 fdim(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_fdim<double4>(x, y);
}
inline double8 fdim(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_fdim<double8>(x, y);
}
inline double16 fdim(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_fdim<double16>(x, y);
}
inline half fdim(half x, half y) __NOEXC {
  return __sycl_std::__invoke_fdim<half>(x, y);
}
inline half2 fdim(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_fdim<half2>(x, y);
}
inline half3 fdim(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_fdim<half3>(x, y);
}
inline half4 fdim(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_fdim<half4>(x, y);
}
inline half8 fdim(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_fdim<half8>(x, y);
}
inline half16 fdim(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_fdim<half16>(x, y);
}

// genfloat floor (genfloat x)
inline float floor(float x) __NOEXC {
  return __sycl_std::__invoke_floor<float>(x);
}
inline float2 floor(float2 x) __NOEXC {
  return __sycl_std::__invoke_floor<float2>(x);
}
inline float3 floor(float3 x) __NOEXC {
  return __sycl_std::__invoke_floor<float3>(x);
}
inline float4 floor(float4 x) __NOEXC {
  return __sycl_std::__invoke_floor<float4>(x);
}
inline float8 floor(float8 x) __NOEXC {
  return __sycl_std::__invoke_floor<float8>(x);
}
inline float16 floor(float16 x) __NOEXC {
  return __sycl_std::__invoke_floor<float16>(x);
}
inline double floor(double x) __NOEXC {
  return __sycl_std::__invoke_floor<double>(x);
}
inline double2 floor(double2 x) __NOEXC {
  return __sycl_std::__invoke_floor<double2>(x);
}
inline double3 floor(double3 x) __NOEXC {
  return __sycl_std::__invoke_floor<double3>(x);
}
inline double4 floor(double4 x) __NOEXC {
  return __sycl_std::__invoke_floor<double4>(x);
}
inline double8 floor(double8 x) __NOEXC {
  return __sycl_std::__invoke_floor<double8>(x);
}
inline double16 floor(double16 x) __NOEXC {
  return __sycl_std::__invoke_floor<double16>(x);
}
inline half floor(half x) __NOEXC {
  return __sycl_std::__invoke_floor<half>(x);
}
inline half2 floor(half2 x) __NOEXC {
  return __sycl_std::__invoke_floor<half2>(x);
}
inline half3 floor(half3 x) __NOEXC {
  return __sycl_std::__invoke_floor<half3>(x);
}
inline half4 floor(half4 x) __NOEXC {
  return __sycl_std::__invoke_floor<half4>(x);
}
inline half8 floor(half8 x) __NOEXC {
  return __sycl_std::__invoke_floor<half8>(x);
}
inline half16 floor(half16 x) __NOEXC {
  return __sycl_std::__invoke_floor<half16>(x);
}

// genfloat fma (genfloat a, genfloat b, genfloat c)
inline float fma(float a, float b, float c) __NOEXC {
  return __sycl_std::__invoke_fma<float>(a, b, c);
}
inline float2 fma(float2 a, float2 b, float2 c) __NOEXC {
  return __sycl_std::__invoke_fma<float2>(a, b, c);
}
inline float3 fma(float3 a, float3 b, float3 c) __NOEXC {
  return __sycl_std::__invoke_fma<float3>(a, b, c);
}
inline float4 fma(float4 a, float4 b, float4 c) __NOEXC {
  return __sycl_std::__invoke_fma<float4>(a, b, c);
}
inline float8 fma(float8 a, float8 b, float8 c) __NOEXC {
  return __sycl_std::__invoke_fma<float8>(a, b, c);
}
inline float16 fma(float16 a, float16 b, float16 c) __NOEXC {
  return __sycl_std::__invoke_fma<float16>(a, b, c);
}
inline double fma(double a, double b, double c) __NOEXC {
  return __sycl_std::__invoke_fma<double>(a, b, c);
}
inline double2 fma(double2 a, double2 b, double2 c) __NOEXC {
  return __sycl_std::__invoke_fma<double2>(a, b, c);
}
inline double3 fma(double3 a, double3 b, double3 c) __NOEXC {
  return __sycl_std::__invoke_fma<double3>(a, b, c);
}
inline double4 fma(double4 a, double4 b, double4 c) __NOEXC {
  return __sycl_std::__invoke_fma<double4>(a, b, c);
}
inline double8 fma(double8 a, double8 b, double8 c) __NOEXC {
  return __sycl_std::__invoke_fma<double8>(a, b, c);
}
inline double16 fma(double16 a, double16 b, double16 c) __NOEXC {
  return __sycl_std::__invoke_fma<double16>(a, b, c);
}
inline half fma(half a, half b, half c) __NOEXC {
  return __sycl_std::__invoke_fma<half>(a, b, c);
}
inline half2 fma(half2 a, half2 b, half2 c) __NOEXC {
  return __sycl_std::__invoke_fma<half2>(a, b, c);
}
inline half3 fma(half3 a, half3 b, half3 c) __NOEXC {
  return __sycl_std::__invoke_fma<half3>(a, b, c);
}
inline half4 fma(half4 a, half4 b, half4 c) __NOEXC {
  return __sycl_std::__invoke_fma<half4>(a, b, c);
}
inline half8 fma(half8 a, half8 b, half8 c) __NOEXC {
  return __sycl_std::__invoke_fma<half8>(a, b, c);
}
inline half16 fma(half16 a, half16 b, half16 c) __NOEXC {
  return __sycl_std::__invoke_fma<half16>(a, b, c);
}

// genfloat fmax (genfloat x, genfloat y)
inline float fmax(float x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float>(x, y);
}
inline float2 fmax(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_fmax<float2>(x, y);
}
inline float3 fmax(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_fmax<float3>(x, y);
}
inline float4 fmax(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_fmax<float4>(x, y);
}
inline float8 fmax(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_fmax<float8>(x, y);
}
inline float16 fmax(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_fmax<float16>(x, y);
}
inline double fmax(double x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double>(x, y);
}
inline double2 fmax(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_fmax<double2>(x, y);
}
inline double3 fmax(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_fmax<double3>(x, y);
}
inline double4 fmax(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_fmax<double4>(x, y);
}
inline double8 fmax(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_fmax<double8>(x, y);
}
inline double16 fmax(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_fmax<double16>(x, y);
}
inline half fmax(half x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half>(x, y);
}
inline half2 fmax(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_fmax<half2>(x, y);
}
inline half3 fmax(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_fmax<half3>(x, y);
}
inline half4 fmax(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_fmax<half4>(x, y);
}
inline half8 fmax(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_fmax<half8>(x, y);
}
inline half16 fmax(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_fmax<half16>(x, y);
}

// genfloat fmax (genfloat x, sgenfloat y)
inline float2 fmax(float2 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float2>(x, float2(y));
}
inline float3 fmax(float3 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float3>(x, float3(y));
}
inline float4 fmax(float4 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float4>(x, float4(y));
}
inline float8 fmax(float8 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float8>(x, float8(y));
}
inline float16 fmax(float16 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmax<float16>(x, float16(y));
}
inline double2 fmax(double2 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double2>(x, double2(y));
}
inline double3 fmax(double3 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double3>(x, double3(y));
}
inline double4 fmax(double4 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double4>(x, double4(y));
}
inline double8 fmax(double8 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double8>(x, double8(y));
}
inline double16 fmax(double16 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmax<double16>(x, double16(y));
}
inline half2 fmax(half2 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half2>(x, half2(y));
}
inline half3 fmax(half3 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half3>(x, half3(y));
}
inline half4 fmax(half4 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half4>(x, half4(y));
}
inline half8 fmax(half8 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half8>(x, half8(y));
}
inline half16 fmax(half16 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmax<half16>(x, half16(y));
}

// genfloat fmin (genfloat x, genfloat y)
inline float fmin(float x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float>(x, y);
}
inline float2 fmin(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float2>(x, y);
}
inline float3 fmin(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float3>(x, y);
}
inline float4 fmin(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float4>(x, y);
}
inline float8 fmin(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float8>(x, y);
}
inline float16 fmin(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float16>(x, y);
}
inline double fmin(double x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double>(x, y);
}
inline double2 fmin(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double2>(x, y);
}
inline double3 fmin(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double3>(x, y);
}
inline double4 fmin(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double4>(x, y);
}
inline double8 fmin(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double8>(x, y);
}
inline double16 fmin(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double16>(x, y);
}
inline half fmin(half x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half>(x, y);
}
inline half2 fmin(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half2>(x, y);
}
inline half3 fmin(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half3>(x, y);
}
inline half4 fmin(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half4>(x, y);
}
inline half8 fmin(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half8>(x, y);
}
inline half16 fmin(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half16>(x, y);
}

// genfloat fmin (genfloat x, sgenfloat y)
inline float2 fmin(float2 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float2>(x, float2(y));
}
inline float3 fmin(float3 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float3>(x, float3(y));
}
inline float4 fmin(float4 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float4>(x, float4(y));
}
inline float8 fmin(float8 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float8>(x, float8(y));
}
inline float16 fmin(float16 x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float16>(x, float16(y));
}
inline double2 fmin(double2 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double2>(x, double2(y));
}
inline double3 fmin(double3 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double3>(x, double3(y));
}
inline double4 fmin(double4 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double4>(x, double4(y));
}
inline double8 fmin(double8 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double8>(x, double8(y));
}
inline double16 fmin(double16 x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double16>(x, double16(y));
}
inline half2 fmin(half2 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half2>(x, half2(y));
}
inline half3 fmin(half3 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half3>(x, half3(y));
}
inline half4 fmin(half4 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half4>(x, half4(y));
}
inline half8 fmin(half8 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half8>(x, half8(y));
}
inline half16 fmin(half16 x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half16>(x, half16(y));
}

// genfloat fmod (genfloat x, genfloat y)
inline float fmod(float x, float y) __NOEXC {
  return __sycl_std::__invoke_fmin<float>(x, y);
}
inline float2 fmod(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float2>(x, y);
}
inline float3 fmod(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float3>(x, y);
}
inline float4 fmod(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float4>(x, y);
}
inline float8 fmod(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float8>(x, y);
}
inline float16 fmod(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<float16>(x, y);
}
inline double fmod(double x, double y) __NOEXC {
  return __sycl_std::__invoke_fmin<double>(x, y);
}
inline double2 fmod(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double2>(x, y);
}
inline double3 fmod(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double3>(x, y);
}
inline double4 fmod(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double4>(x, y);
}
inline double8 fmod(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double8>(x, y);
}
inline double16 fmod(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<double16>(x, y);
}
inline half fmod(half x, half y) __NOEXC {
  return __sycl_std::__invoke_fmin<half>(x, y);
}
inline half2 fmod(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half2>(x, y);
}
inline half3 fmod(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half3>(x, y);
}
inline half4 fmod(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half4>(x, y);
}
inline half8 fmod(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half8>(x, y);
}
inline half16 fmod(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_fmin<half16>(x, y);
}

// genfloat fract (genfloat x, genfloatptr iptr)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float fract(float x,
                   multi_ptr<float, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2 fract(float2 x,
                    multi_ptr<float2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3 fract(float3 x,
                    multi_ptr<float3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4 fract(float4 x,
                    multi_ptr<float4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8 fract(float8 x,
                    multi_ptr<float8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16
fract(float16 x, multi_ptr<float16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double fract(double x,
                    multi_ptr<double, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2
fract(double2 x, multi_ptr<double2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3
fract(double3 x, multi_ptr<double3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4
fract(double4 x, multi_ptr<double4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8
fract(double8 x, multi_ptr<double8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16
fract(double16 x, multi_ptr<double16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half fract(half x,
                  multi_ptr<half, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2 fract(half2 x,
                   multi_ptr<half2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3 fract(half3 x,
                   multi_ptr<half3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4 fract(half4 x,
                   multi_ptr<half4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8 fract(half8 x,
                   multi_ptr<half8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16 fract(half16 x,
                    multi_ptr<half16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half16>(x, iptr);
}

// genfloat frexp(genfloat x, genintptr exp)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float frexp(float x,
                   multi_ptr<int, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<int>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2 frexp(float2 x,
                    multi_ptr<int2, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<float2>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3 frexp(float3 x,
                    multi_ptr<int3, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<float3>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4 frexp(float4 x,
                    multi_ptr<int4, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<float4>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8 frexp(float8 x,
                    multi_ptr<int8, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<float8>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16 frexp(float16 x,
                     multi_ptr<int16, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<float16>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double frexp(double x,
                    multi_ptr<int, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2 frexp(double2 x,
                     multi_ptr<int2, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double2>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3 frexp(double3 x,
                     multi_ptr<int3, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double3>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4 frexp(double4 x,
                     multi_ptr<int4, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double4>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8 frexp(double8 x,
                     multi_ptr<int8, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double8>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16 frexp(double16 x,
                      multi_ptr<int16, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<double16>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half frexp(half x,
                  multi_ptr<int, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2 frexp(half2 x,
                   multi_ptr<int2, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half2>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3 frexp(half3 x,
                   multi_ptr<int3, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half3>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4 frexp(half4 x,
                   multi_ptr<int4, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half4>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8 frexp(half8 x,
                   multi_ptr<int8, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half8>(x, exp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16 frexp(half16 x,
                    multi_ptr<int16, AddressSpace, IsDecorated> exp) __NOEXC {
  return __sycl_std::__invoke_frexp<half16>(x, exp);
}

// genfloat hypot (genfloat x, genfloat y)
inline float hypot(float x, float y) __NOEXC {
  return __sycl_std::__invoke_hypot<float>(x, y);
}
inline float2 hypot(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_hypot<float2>(x, y);
}
inline float3 hypot(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_hypot<float3>(x, y);
}
inline float4 hypot(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_hypot<float4>(x, y);
}
inline float8 hypot(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_hypot<float8>(x, y);
}
inline float16 hypot(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_hypot<float16>(x, y);
}
inline double hypot(double x, double y) __NOEXC {
  return __sycl_std::__invoke_hypot<double>(x, y);
}
inline double2 hypot(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_hypot<double2>(x, y);
}
inline double3 hypot(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_hypot<double3>(x, y);
}
inline double4 hypot(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_hypot<double4>(x, y);
}
inline double8 hypot(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_hypot<double8>(x, y);
}
inline double16 hypot(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_hypot<double16>(x, y);
}
inline half hypot(half x, half y) __NOEXC {
  return __sycl_std::__invoke_hypot<half>(x, y);
}
inline half2 hypot(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_hypot<half2>(x, y);
}
inline half3 hypot(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_hypot<half3>(x, y);
}
inline half4 hypot(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_hypot<half4>(x, y);
}
inline half8 hypot(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_hypot<half8>(x, y);
}
inline half16 hypot(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_hypot<half16>(x, y);
}

// genint ilogb (genfloat x)
inline int ilogb(float x) __NOEXC { return __sycl_std::__invoke_ilogb<int>(x); }
inline int2 ilogb(float2 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int2>(x);
}
inline int3 ilogb(float3 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int3>(x);
}
inline int4 ilogb(float4 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int4>(x);
}
inline int8 ilogb(float8 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int8>(x);
}
inline int16 ilogb(float16 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int16>(x);
}
inline int ilogb(double x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int>(x);
}
inline int2 ilogb(double2 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int2>(x);
}
inline int3 ilogb(double3 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int3>(x);
}
inline int4 ilogb(double4 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int4>(x);
}
inline int8 ilogb(double8 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int8>(x);
}
inline int16 ilogb(double16 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int16>(x);
}
inline int ilogb(half x) __NOEXC { return __sycl_std::__invoke_ilogb<int>(x); }
inline int2 ilogb(half2 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int2>(x);
}
inline int3 ilogb(half3 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int3>(x);
}
inline int4 ilogb(half4 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int4>(x);
}
inline int8 ilogb(half8 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int8>(x);
}
inline int16 ilogb(half16 x) __NOEXC {
  return __sycl_std::__invoke_ilogb<int16>(x);
}

// genfloat ldexp (genfloat x, genint k)
inline float ldexp(float x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float>(x, k);
}
inline float2 ldexp(float2 x, int2 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float2>(x, k);
}
inline float3 ldexp(float3 x, int3 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float3>(x, k);
}
inline float4 ldexp(float4 x, int4 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float4>(x, k);
}
inline float8 ldexp(float8 x, int8 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float8>(x, k);
}
inline float16 ldexp(float16 x, int16 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float16>(x, k);
}
inline double ldexp(double x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double>(x, k);
}
inline double2 ldexp(double2 x, int2 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double2>(x, k);
}
inline double3 ldexp(double3 x, int3 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double3>(x, k);
}
inline double4 ldexp(double4 x, int4 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double4>(x, k);
}
inline double8 ldexp(double8 x, int8 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double8>(x, k);
}
inline double16 ldexp(double16 x, int16 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double16>(x, k);
}
inline half ldexp(half x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half>(x, k);
}
inline half2 ldexp(half2 x, int2 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half2>(x, k);
}
inline half3 ldexp(half3 x, int3 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half3>(x, k);
}
inline half4 ldexp(half4 x, int4 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half4>(x, k);
}
inline half8 ldexp(half8 x, int8 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half8>(x, k);
}
inline half16 ldexp(half16 x, int16 k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half16>(x, k);
}

// genfloat ldexp (genfloat x, int k)
inline float2 ldexp(float2 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float2>(x, int2(k));
}
inline float3 ldexp(float3 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float3>(x, int3(k));
}
inline float4 ldexp(float4 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float4>(x, int4(k));
}
inline float8 ldexp(float8 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float8>(x, int8(k));
}
inline float16 ldexp(float16 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<float16>(x, int16(k));
}
inline double2 ldexp(double2 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double2>(x, int2(k));
}
inline double3 ldexp(double3 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double3>(x, int3(k));
}
inline double4 ldexp(double4 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double4>(x, int4(k));
}
inline double8 ldexp(double8 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double8>(x, int8(k));
}
inline double16 ldexp(double16 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<double16>(x, int16(k));
}
inline half2 ldexp(half2 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half2>(x, int2(k));
}
inline half3 ldexp(half3 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half3>(x, int3(k));
}
inline half4 ldexp(half4 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half4>(x, int4(k));
}
inline half8 ldexp(half8 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half8>(x, int8(k));
}
inline half16 ldexp(half16 x, int k) __NOEXC {
  return __sycl_std::__invoke_ldexp<half16>(x, int16(k));
}

// genfloat lgamma (genfloat x)
inline float lgamma(float x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float>(x);
}
inline float2 lgamma(float2 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float2>(x);
}
inline float3 lgamma(float3 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float3>(x);
}
inline float4 lgamma(float4 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float4>(x);
}
inline float8 lgamma(float8 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float8>(x);
}
inline float16 lgamma(float16 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<float16>(x);
}
inline double lgamma(double x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double>(x);
}
inline double2 lgamma(double2 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double2>(x);
}
inline double3 lgamma(double3 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double3>(x);
}
inline double4 lgamma(double4 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double4>(x);
}
inline double8 lgamma(double8 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double8>(x);
}
inline double16 lgamma(double16 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<double16>(x);
}
inline half lgamma(half x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half>(x);
}
inline half2 lgamma(half2 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half2>(x);
}
inline half3 lgamma(half3 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half3>(x);
}
inline half4 lgamma(half4 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half4>(x);
}
inline half8 lgamma(half8 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half8>(x);
}
inline half16 lgamma(half16 x) __NOEXC {
  return __sycl_std::__invoke_lgamma<half16>(x);
}

// genfloat lgamma_r (genfloat x, genintptr signp)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float lgamma_r(float x,
                      multi_ptr<int, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2
lgamma_r(float2 x, multi_ptr<int2, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float2>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3
lgamma_r(float3 x, multi_ptr<int3, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float3>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4
lgamma_r(float4 x, multi_ptr<int4, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float4>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8
lgamma_r(float8 x, multi_ptr<int8, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float8>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16
lgamma_r(float16 x, multi_ptr<int16, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<float16>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double
lgamma_r(double x, multi_ptr<double, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2
lgamma_r(double2 x,
         multi_ptr<double2, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double2>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3
lgamma_r(double3 x,
         multi_ptr<double3, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double3>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4
lgamma_r(double4 x,
         multi_ptr<double4, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double4>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8
lgamma_r(double8 x,
         multi_ptr<double8, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double8>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16
lgamma_r(double16 x,
         multi_ptr<double16, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<double16>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half lgamma_r(half x,
                     multi_ptr<half, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2
lgamma_r(half2 x, multi_ptr<half2, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half2>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3
lgamma_r(half3 x, multi_ptr<half3, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half3>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4
lgamma_r(half4 x, multi_ptr<half4, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half4>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8
lgamma_r(half8 x, multi_ptr<half8, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half8>(x, signp);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16
lgamma_r(half16 x, multi_ptr<half16, AddressSpace, IsDecorated> signp) __NOEXC {
  return __sycl_std::__invoke_lgamma_r<half16>(x, signp);
}

// genfloat log (genfloat x)
inline float log(float x) __NOEXC { return __sycl_std::__invoke_log<float>(x); }
inline float2 log(float2 x) __NOEXC {
  return __sycl_std::__invoke_log<float2>(x);
}
inline float3 log(float3 x) __NOEXC {
  return __sycl_std::__invoke_log<float3>(x);
}
inline float4 log(float4 x) __NOEXC {
  return __sycl_std::__invoke_log<float4>(x);
}
inline float8 log(float8 x) __NOEXC {
  return __sycl_std::__invoke_log<float8>(x);
}
inline float16 log(float16 x) __NOEXC {
  return __sycl_std::__invoke_log<float16>(x);
}
inline double log(double x) __NOEXC {
  return __sycl_std::__invoke_log<double>(x);
}
inline double2 log(double2 x) __NOEXC {
  return __sycl_std::__invoke_log<double2>(x);
}
inline double3 log(double3 x) __NOEXC {
  return __sycl_std::__invoke_log<double3>(x);
}
inline double4 log(double4 x) __NOEXC {
  return __sycl_std::__invoke_log<double4>(x);
}
inline double8 log(double8 x) __NOEXC {
  return __sycl_std::__invoke_log<double8>(x);
}
inline double16 log(double16 x) __NOEXC {
  return __sycl_std::__invoke_log<double16>(x);
}
inline half log(half x) __NOEXC { return __sycl_std::__invoke_log<half>(x); }
inline half2 log(half2 x) __NOEXC { return __sycl_std::__invoke_log<half2>(x); }
inline half3 log(half3 x) __NOEXC { return __sycl_std::__invoke_log<half3>(x); }
inline half4 log(half4 x) __NOEXC { return __sycl_std::__invoke_log<half4>(x); }
inline half8 log(half8 x) __NOEXC { return __sycl_std::__invoke_log<half8>(x); }
inline half16 log(half16 x) __NOEXC {
  return __sycl_std::__invoke_log<half16>(x);
}

// genfloat log2 (genfloat x)
inline float log2(float x) __NOEXC {
  return __sycl_std::__invoke_log2<float>(x);
}
inline float2 log2(float2 x) __NOEXC {
  return __sycl_std::__invoke_log2<float2>(x);
}
inline float3 log2(float3 x) __NOEXC {
  return __sycl_std::__invoke_log2<float3>(x);
}
inline float4 log2(float4 x) __NOEXC {
  return __sycl_std::__invoke_log2<float4>(x);
}
inline float8 log2(float8 x) __NOEXC {
  return __sycl_std::__invoke_log2<float8>(x);
}
inline float16 log2(float16 x) __NOEXC {
  return __sycl_std::__invoke_log2<float16>(x);
}
inline double log2(double x) __NOEXC {
  return __sycl_std::__invoke_log2<double>(x);
}
inline double2 log2(double2 x) __NOEXC {
  return __sycl_std::__invoke_log2<double2>(x);
}
inline double3 log2(double3 x) __NOEXC {
  return __sycl_std::__invoke_log2<double3>(x);
}
inline double4 log2(double4 x) __NOEXC {
  return __sycl_std::__invoke_log2<double4>(x);
}
inline double8 log2(double8 x) __NOEXC {
  return __sycl_std::__invoke_log2<double8>(x);
}
inline double16 log2(double16 x) __NOEXC {
  return __sycl_std::__invoke_log2<double16>(x);
}
inline half log2(half x) __NOEXC { return __sycl_std::__invoke_log2<half>(x); }
inline half2 log2(half2 x) __NOEXC {
  return __sycl_std::__invoke_log2<half2>(x);
}
inline half3 log2(half3 x) __NOEXC {
  return __sycl_std::__invoke_log2<half3>(x);
}
inline half4 log2(half4 x) __NOEXC {
  return __sycl_std::__invoke_log2<half4>(x);
}
inline half8 log2(half8 x) __NOEXC {
  return __sycl_std::__invoke_log2<half8>(x);
}
inline half16 log2(half16 x) __NOEXC {
  return __sycl_std::__invoke_log2<half16>(x);
}

// genfloat log10 (genfloat x)
inline float log10(float x) __NOEXC {
  return __sycl_std::__invoke_log10<float>(x);
}
inline float2 log10(float2 x) __NOEXC {
  return __sycl_std::__invoke_log10<float2>(x);
}
inline float3 log10(float3 x) __NOEXC {
  return __sycl_std::__invoke_log10<float3>(x);
}
inline float4 log10(float4 x) __NOEXC {
  return __sycl_std::__invoke_log10<float4>(x);
}
inline float8 log10(float8 x) __NOEXC {
  return __sycl_std::__invoke_log10<float8>(x);
}
inline float16 log10(float16 x) __NOEXC {
  return __sycl_std::__invoke_log10<float16>(x);
}
inline double log10(double x) __NOEXC {
  return __sycl_std::__invoke_log10<double>(x);
}
inline double2 log10(double2 x) __NOEXC {
  return __sycl_std::__invoke_log10<double2>(x);
}
inline double3 log10(double3 x) __NOEXC {
  return __sycl_std::__invoke_log10<double3>(x);
}
inline double4 log10(double4 x) __NOEXC {
  return __sycl_std::__invoke_log10<double4>(x);
}
inline double8 log10(double8 x) __NOEXC {
  return __sycl_std::__invoke_log10<double8>(x);
}
inline double16 log10(double16 x) __NOEXC {
  return __sycl_std::__invoke_log10<double16>(x);
}
inline half log10(half x) __NOEXC {
  return __sycl_std::__invoke_log10<half>(x);
}
inline half2 log10(half2 x) __NOEXC {
  return __sycl_std::__invoke_log10<half2>(x);
}
inline half3 log10(half3 x) __NOEXC {
  return __sycl_std::__invoke_log10<half3>(x);
}
inline half4 log10(half4 x) __NOEXC {
  return __sycl_std::__invoke_log10<half4>(x);
}
inline half8 log10(half8 x) __NOEXC {
  return __sycl_std::__invoke_log10<half8>(x);
}
inline half16 log10(half16 x) __NOEXC {
  return __sycl_std::__invoke_log10<half16>(x);
}

// genfloat log1p (genfloat x)
inline float log1p(float x) __NOEXC {
  return __sycl_std::__invoke_log1p<float>(x);
}
inline float2 log1p(float2 x) __NOEXC {
  return __sycl_std::__invoke_log1p<float2>(x);
}
inline float3 log1p(float3 x) __NOEXC {
  return __sycl_std::__invoke_log1p<float3>(x);
}
inline float4 log1p(float4 x) __NOEXC {
  return __sycl_std::__invoke_log1p<float4>(x);
}
inline float8 log1p(float8 x) __NOEXC {
  return __sycl_std::__invoke_log1p<float8>(x);
}
inline float16 log1p(float16 x) __NOEXC {
  return __sycl_std::__invoke_log1p<float16>(x);
}
inline double log1p(double x) __NOEXC {
  return __sycl_std::__invoke_log1p<double>(x);
}
inline double2 log1p(double2 x) __NOEXC {
  return __sycl_std::__invoke_log1p<double2>(x);
}
inline double3 log1p(double3 x) __NOEXC {
  return __sycl_std::__invoke_log1p<double3>(x);
}
inline double4 log1p(double4 x) __NOEXC {
  return __sycl_std::__invoke_log1p<double4>(x);
}
inline double8 log1p(double8 x) __NOEXC {
  return __sycl_std::__invoke_log1p<double8>(x);
}
inline double16 log1p(double16 x) __NOEXC {
  return __sycl_std::__invoke_log1p<double16>(x);
}
inline half log1p(half x) __NOEXC {
  return __sycl_std::__invoke_log1p<half>(x);
}
inline half2 log1p(half2 x) __NOEXC {
  return __sycl_std::__invoke_log1p<half2>(x);
}
inline half3 log1p(half3 x) __NOEXC {
  return __sycl_std::__invoke_log1p<half3>(x);
}
inline half4 log1p(half4 x) __NOEXC {
  return __sycl_std::__invoke_log1p<half4>(x);
}
inline half8 log1p(half8 x) __NOEXC {
  return __sycl_std::__invoke_log1p<half8>(x);
}
inline half16 log1p(half16 x) __NOEXC {
  return __sycl_std::__invoke_log1p<half16>(x);
}

// genfloat logb (genfloat x)
inline float logb(float x) __NOEXC {
  return __sycl_std::__invoke_logb<float>(x);
}
inline float2 logb(float2 x) __NOEXC {
  return __sycl_std::__invoke_logb<float2>(x);
}
inline float3 logb(float3 x) __NOEXC {
  return __sycl_std::__invoke_logb<float3>(x);
}
inline float4 logb(float4 x) __NOEXC {
  return __sycl_std::__invoke_logb<float4>(x);
}
inline float8 logb(float8 x) __NOEXC {
  return __sycl_std::__invoke_logb<float8>(x);
}
inline float16 logb(float16 x) __NOEXC {
  return __sycl_std::__invoke_logb<float16>(x);
}
inline double logb(double x) __NOEXC {
  return __sycl_std::__invoke_logb<double>(x);
}
inline double2 logb(double2 x) __NOEXC {
  return __sycl_std::__invoke_logb<double2>(x);
}
inline double3 logb(double3 x) __NOEXC {
  return __sycl_std::__invoke_logb<double3>(x);
}
inline double4 logb(double4 x) __NOEXC {
  return __sycl_std::__invoke_logb<double4>(x);
}
inline double8 logb(double8 x) __NOEXC {
  return __sycl_std::__invoke_logb<double8>(x);
}
inline double16 logb(double16 x) __NOEXC {
  return __sycl_std::__invoke_logb<double16>(x);
}
inline half logb(half x) __NOEXC { return __sycl_std::__invoke_logb<half>(x); }
inline half2 logb(half2 x) __NOEXC {
  return __sycl_std::__invoke_logb<half2>(x);
}
inline half3 logb(half3 x) __NOEXC {
  return __sycl_std::__invoke_logb<half3>(x);
}
inline half4 logb(half4 x) __NOEXC {
  return __sycl_std::__invoke_logb<half4>(x);
}
inline half8 logb(half8 x) __NOEXC {
  return __sycl_std::__invoke_logb<half8>(x);
}
inline half16 logb(half16 x) __NOEXC {
  return __sycl_std::__invoke_logb<half16>(x);
}

// genfloat mad (genfloat a, genfloat b, genfloat c)
inline float mad(float a, float b, float c) __NOEXC {
  return __sycl_std::__invoke_mad<float>(a, b, c);
}
inline float2 mad(float2 a, float2 b, float2 c) __NOEXC {
  return __sycl_std::__invoke_mad<float2>(a, b, c);
}
inline float3 mad(float3 a, float3 b, float3 c) __NOEXC {
  return __sycl_std::__invoke_mad<float3>(a, b, c);
}
inline float4 mad(float4 a, float4 b, float4 c) __NOEXC {
  return __sycl_std::__invoke_mad<float4>(a, b, c);
}
inline float8 mad(float8 a, float8 b, float8 c) __NOEXC {
  return __sycl_std::__invoke_mad<float8>(a, b, c);
}
inline float16 mad(float16 a, float16 b, float16 c) __NOEXC {
  return __sycl_std::__invoke_mad<float16>(a, b, c);
}
inline double mad(double a, double b, double c) __NOEXC {
  return __sycl_std::__invoke_mad<double>(a, b, c);
}
inline double2 mad(double2 a, double2 b, double2 c) __NOEXC {
  return __sycl_std::__invoke_mad<double2>(a, b, c);
}
inline double3 mad(double3 a, double3 b, double3 c) __NOEXC {
  return __sycl_std::__invoke_mad<double3>(a, b, c);
}
inline double4 mad(double4 a, double4 b, double4 c) __NOEXC {
  return __sycl_std::__invoke_mad<double4>(a, b, c);
}
inline double8 mad(double8 a, double8 b, double8 c) __NOEXC {
  return __sycl_std::__invoke_mad<double8>(a, b, c);
}
inline double16 mad(double16 a, double16 b, double16 c) __NOEXC {
  return __sycl_std::__invoke_mad<double16>(a, b, c);
}
inline half mad(half a, half b, half c) __NOEXC {
  return __sycl_std::__invoke_mad<half>(a, b, c);
}
inline half2 mad(half2 a, half2 b, half2 c) __NOEXC {
  return __sycl_std::__invoke_mad<half2>(a, b, c);
}
inline half3 mad(half3 a, half3 b, half3 c) __NOEXC {
  return __sycl_std::__invoke_mad<half3>(a, b, c);
}
inline half4 mad(half4 a, half4 b, half4 c) __NOEXC {
  return __sycl_std::__invoke_mad<half4>(a, b, c);
}
inline half8 mad(half8 a, half8 b, half8 c) __NOEXC {
  return __sycl_std::__invoke_mad<half8>(a, b, c);
}
inline half16 mad(half16 a, half16 b, half16 c) __NOEXC {
  return __sycl_std::__invoke_mad<half16>(a, b, c);
}

// genfloat maxmag (genfloat x, genfloat y)
inline float maxmag(float x, float y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float>(x, y);
}
inline float2 maxmag(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float2>(x, y);
}
inline float3 maxmag(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float3>(x, y);
}
inline float4 maxmag(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float4>(x, y);
}
inline float8 maxmag(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float8>(x, y);
}
inline float16 maxmag(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<float16>(x, y);
}
inline double maxmag(double x, double y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double>(x, y);
}
inline double2 maxmag(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double2>(x, y);
}
inline double3 maxmag(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double3>(x, y);
}
inline double4 maxmag(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double4>(x, y);
}
inline double8 maxmag(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double8>(x, y);
}
inline double16 maxmag(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<double16>(x, y);
}
inline half maxmag(half x, half y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half>(x, y);
}
inline half2 maxmag(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half2>(x, y);
}
inline half3 maxmag(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half3>(x, y);
}
inline half4 maxmag(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half4>(x, y);
}
inline half8 maxmag(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half8>(x, y);
}
inline half16 maxmag(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_maxmag<half16>(x, y);
}

// genfloat minmag (genfloat x, genfloat y)
inline float minmag(float x, float y) __NOEXC {
  return __sycl_std::__invoke_minmag<float>(x, y);
}
inline float2 minmag(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_minmag<float2>(x, y);
}
inline float3 minmag(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_minmag<float3>(x, y);
}
inline float4 minmag(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_minmag<float4>(x, y);
}
inline float8 minmag(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_minmag<float8>(x, y);
}
inline float16 minmag(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_minmag<float16>(x, y);
}
inline double minmag(double x, double y) __NOEXC {
  return __sycl_std::__invoke_minmag<double>(x, y);
}
inline double2 minmag(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_minmag<double2>(x, y);
}
inline double3 minmag(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_minmag<double3>(x, y);
}
inline double4 minmag(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_minmag<double4>(x, y);
}
inline double8 minmag(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_minmag<double8>(x, y);
}
inline double16 minmag(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_minmag<double16>(x, y);
}
inline half minmag(half x, half y) __NOEXC {
  return __sycl_std::__invoke_minmag<half>(x, y);
}
inline half2 minmag(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_minmag<half2>(x, y);
}
inline half3 minmag(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_minmag<half3>(x, y);
}
inline half4 minmag(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_minmag<half4>(x, y);
}
inline half8 minmag(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_minmag<half8>(x, y);
}
inline half16 minmag(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_minmag<half16>(x, y);
}

// genfloat modf (genfloat x, genfloatptr iptr)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float modf(float x,
                  multi_ptr<float, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2 modf(float2 x,
                   multi_ptr<float2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3 modf(float3 x,
                   multi_ptr<float3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4 modf(float4 x,
                   multi_ptr<float4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8 modf(float8 x,
                   multi_ptr<float8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16
modf(float16 x, multi_ptr<float16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double modf(double x,
                   multi_ptr<double, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2
modf(double2 x, multi_ptr<double2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3
modf(double3 x, multi_ptr<double3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4
modf(double4 x, multi_ptr<double4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8
modf(double8 x, multi_ptr<double8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16
modf(double16 x, multi_ptr<double16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half modf(half x,
                 multi_ptr<half, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2 modf(half2 x,
                  multi_ptr<half2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3 modf(half3 x,
                  multi_ptr<half3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4 modf(half4 x,
                  multi_ptr<half4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8 modf(half8 x,
                  multi_ptr<half8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16 modf(half16 x,
                   multi_ptr<half16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half16>(x, iptr);
}

// genfloatf nan(ugenint nancode)
inline float nan(unsigned int nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float>(nancode);
}
inline float2 nan(uint2 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float2>(nancode);
}
inline float3 nan(uint3 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float3>(nancode);
}
inline float4 nan(uint4 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float4>(nancode);
}
inline float8 nan(uint8 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float8>(nancode);
}
inline float16 nan(uint16 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<float16>(nancode);
}
// genfloatd nan(ugenlonginteger nancode)
inline double nan(unsigned long nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double>(nancode);
}
inline double2 nan(ulong2 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double2>(nancode);
}
inline double3 nan(ulong3 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double3>(nancode);
}
inline double4 nan(ulong4 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double4>(nancode);
}
inline double8 nan(ulong8 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double8>(nancode);
}
inline double16 nan(ulong16 nancode) __NOEXC {
  return __sycl_std::__invoke_nan<double16>(nancode);
}

// genfloat nextafter (genfloat x, genfloat y)
inline float nextafter(float x, float y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float>(x, y);
}
inline float2 nextafter(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float2>(x, y);
}
inline float3 nextafter(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float3>(x, y);
}
inline float4 nextafter(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float4>(x, y);
}
inline float8 nextafter(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float8>(x, y);
}
inline float16 nextafter(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<float16>(x, y);
}
inline double nextafter(double x, double y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double>(x, y);
}
inline double2 nextafter(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double2>(x, y);
}
inline double3 nextafter(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double3>(x, y);
}
inline double4 nextafter(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double4>(x, y);
}
inline double8 nextafter(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double8>(x, y);
}
inline double16 nextafter(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<double16>(x, y);
}
inline half nextafter(half x, half y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half>(x, y);
}
inline half2 nextafter(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half2>(x, y);
}
inline half3 nextafter(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half3>(x, y);
}
inline half4 nextafter(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half4>(x, y);
}
inline half8 nextafter(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half8>(x, y);
}
inline half16 nextafter(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_nextafter<half16>(x, y);
}

// genfloat pow (genfloat x, genfloat y)
inline float pow(float x, float y) __NOEXC {
  return __sycl_std::__invoke_pow<float>(x, y);
}
inline float2 pow(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_pow<float2>(x, y);
}
inline float3 pow(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_pow<float3>(x, y);
}
inline float4 pow(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_pow<float4>(x, y);
}
inline float8 pow(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_pow<float8>(x, y);
}
inline float16 pow(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_pow<float16>(x, y);
}
inline double pow(double x, double y) __NOEXC {
  return __sycl_std::__invoke_pow<double>(x, y);
}
inline double2 pow(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_pow<double2>(x, y);
}
inline double3 pow(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_pow<double3>(x, y);
}
inline double4 pow(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_pow<double4>(x, y);
}
inline double8 pow(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_pow<double8>(x, y);
}
inline double16 pow(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_pow<double16>(x, y);
}
inline half pow(half x, half y) __NOEXC {
  return __sycl_std::__invoke_pow<half>(x, y);
}
inline half2 pow(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_pow<half2>(x, y);
}
inline half3 pow(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_pow<half3>(x, y);
}
inline half4 pow(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_pow<half4>(x, y);
}
inline half8 pow(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_pow<half8>(x, y);
}
inline half16 pow(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_pow<half16>(x, y);
}

// genfloat pown (genfloat x, genint y)
inline float pown(float x, int y) __NOEXC {
  return __sycl_std::__invoke_pown<float>(x, y);
}
inline float2 pown(float2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_pown<float2>(x, y);
}
inline float3 pown(float3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_pown<float3>(x, y);
}
inline float4 pown(float4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_pown<float4>(x, y);
}
inline float8 pown(float8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_pown<float8>(x, y);
}
inline float16 pown(float16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_pown<float16>(x, y);
}
inline double pown(double x, int y) __NOEXC {
  return __sycl_std::__invoke_pown<double>(x, y);
}
inline double2 pown(double2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_pown<double2>(x, y);
}
inline double3 pown(double3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_pown<double3>(x, y);
}
inline double4 pown(double4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_pown<double4>(x, y);
}
inline double8 pown(double8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_pown<double8>(x, y);
}
inline double16 pown(double16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_pown<double16>(x, y);
}
inline half pown(half x, int y) __NOEXC {
  return __sycl_std::__invoke_pown<half>(x, y);
}
inline half2 pown(half2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_pown<half2>(x, y);
}
inline half3 pown(half3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_pown<half3>(x, y);
}
inline half4 pown(half4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_pown<half4>(x, y);
}
inline half8 pown(half8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_pown<half8>(x, y);
}
inline half16 pown(half16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_pown<half16>(x, y);
}

// genfloat powr (genfloat x, genfloat y)
inline float powr(float x, float y) __NOEXC {
  return __sycl_std::__invoke_powr<float>(x, y);
}
inline float2 powr(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_powr<float2>(x, y);
}
inline float3 powr(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_powr<float3>(x, y);
}
inline float4 powr(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_powr<float4>(x, y);
}
inline float8 powr(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_powr<float8>(x, y);
}
inline float16 powr(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_powr<float16>(x, y);
}
inline double powr(double x, double y) __NOEXC {
  return __sycl_std::__invoke_powr<double>(x, y);
}
inline double2 powr(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_powr<double2>(x, y);
}
inline double3 powr(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_powr<double3>(x, y);
}
inline double4 powr(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_powr<double4>(x, y);
}
inline double8 powr(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_powr<double8>(x, y);
}
inline double16 powr(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_powr<double16>(x, y);
}
inline half powr(half x, half y) __NOEXC {
  return __sycl_std::__invoke_powr<half>(x, y);
}
inline half2 powr(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_powr<half2>(x, y);
}
inline half3 powr(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_powr<half3>(x, y);
}
inline half4 powr(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_powr<half4>(x, y);
}
inline half8 powr(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_powr<half8>(x, y);
}
inline half16 powr(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_powr<half16>(x, y);
}

// genfloat remainder(genfloat x, genfloat y)
inline float remainder(float x, float y) __NOEXC {
  return __sycl_std::__invoke_remainder<float>(x, y);
}
inline float2 remainder(float2 x, float2 y) __NOEXC {
  return __sycl_std::__invoke_remainder<float2>(x, y);
}
inline float3 remainder(float3 x, float3 y) __NOEXC {
  return __sycl_std::__invoke_remainder<float3>(x, y);
}
inline float4 remainder(float4 x, float4 y) __NOEXC {
  return __sycl_std::__invoke_remainder<float4>(x, y);
}
inline float8 remainder(float8 x, float8 y) __NOEXC {
  return __sycl_std::__invoke_remainder<float8>(x, y);
}
inline float16 remainder(float16 x, float16 y) __NOEXC {
  return __sycl_std::__invoke_remainder<float16>(x, y);
}
inline double remainder(double x, double y) __NOEXC {
  return __sycl_std::__invoke_remainder<double>(x, y);
}
inline double2 remainder(double2 x, double2 y) __NOEXC {
  return __sycl_std::__invoke_remainder<double2>(x, y);
}
inline double3 remainder(double3 x, double3 y) __NOEXC {
  return __sycl_std::__invoke_remainder<double3>(x, y);
}
inline double4 remainder(double4 x, double4 y) __NOEXC {
  return __sycl_std::__invoke_remainder<double4>(x, y);
}
inline double8 remainder(double8 x, double8 y) __NOEXC {
  return __sycl_std::__invoke_remainder<double8>(x, y);
}
inline double16 remainder(double16 x, double16 y) __NOEXC {
  return __sycl_std::__invoke_remainder<double16>(x, y);
}
inline half remainder(half x, half y) __NOEXC {
  return __sycl_std::__invoke_remainder<half>(x, y);
}
inline half2 remainder(half2 x, half2 y) __NOEXC {
  return __sycl_std::__invoke_remainder<half2>(x, y);
}
inline half3 remainder(half3 x, half3 y) __NOEXC {
  return __sycl_std::__invoke_remainder<half3>(x, y);
}
inline half4 remainder(half4 x, half4 y) __NOEXC {
  return __sycl_std::__invoke_remainder<half4>(x, y);
}
inline half8 remainder(half8 x, half8 y) __NOEXC {
  return __sycl_std::__invoke_remainder<half8>(x, y);
}
inline half16 remainder(half16 x, half16 y) __NOEXC {
  return __sycl_std::__invoke_remainder<half16>(x, y);
}

// svgenfloat remquo (svgenfloat x, svgenfloat y, genintptr quo)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float modf(float x, float y,
                  multi_ptr<int, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2 modf(float2 x, float2 y,
                   multi_ptr<int2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3 modf(float3 x, float3 y,
                   multi_ptr<int3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4 modf(float4 x, float4 y,
                   multi_ptr<int4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8 modf(float8 x, float8 y,
                   multi_ptr<int8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16 modf(float16 x, float16 y,
                    multi_ptr<int16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<float16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double modf(double x, double y,
                   multi_ptr<double, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2
modf(double2 x, double2 y,
     multi_ptr<double2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3
modf(double3 x, double3 y,
     multi_ptr<double3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4
modf(double4 x, double4 y,
     multi_ptr<double4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8
modf(double8 x, double8 y,
     multi_ptr<double8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16
modf(double16 x, double16 y,
     multi_ptr<double16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<double16>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half modf(half x, half y,
                 multi_ptr<half, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2 modf(half2 x, half2 y,
                  multi_ptr<half2, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half2>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3 modf(half3 x, half3 y,
                  multi_ptr<half3, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half3>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4 modf(half4 x, half4 y,
                  multi_ptr<half4, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half4>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8 modf(half8 x, half8 y,
                  multi_ptr<half8, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half8>(x, iptr);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16 modf(half16 x, half16 y,
                   multi_ptr<half16, AddressSpace, IsDecorated> iptr) __NOEXC {
  return __sycl_std::__invoke_fract<half16>(x, iptr);
}

// genfloat rint (genfloat x)
inline float rint(float x) __NOEXC {
  return __sycl_std::__invoke_rint<float>(x);
}
inline float2 rint(float2 x) __NOEXC {
  return __sycl_std::__invoke_rint<float2>(x);
}
inline float3 rint(float3 x) __NOEXC {
  return __sycl_std::__invoke_rint<float3>(x);
}
inline float4 rint(float4 x) __NOEXC {
  return __sycl_std::__invoke_rint<float4>(x);
}
inline float8 rint(float8 x) __NOEXC {
  return __sycl_std::__invoke_rint<float8>(x);
}
inline float16 rint(float16 x) __NOEXC {
  return __sycl_std::__invoke_rint<float16>(x);
}
inline double rint(double x) __NOEXC {
  return __sycl_std::__invoke_rint<double>(x);
}
inline double2 rint(double2 x) __NOEXC {
  return __sycl_std::__invoke_rint<double2>(x);
}
inline double3 rint(double3 x) __NOEXC {
  return __sycl_std::__invoke_rint<double3>(x);
}
inline double4 rint(double4 x) __NOEXC {
  return __sycl_std::__invoke_rint<double4>(x);
}
inline double8 rint(double8 x) __NOEXC {
  return __sycl_std::__invoke_rint<double8>(x);
}
inline double16 rint(double16 x) __NOEXC {
  return __sycl_std::__invoke_rint<double16>(x);
}
inline half rint(half x) __NOEXC { return __sycl_std::__invoke_rint<half>(x); }
inline half2 rint(half2 x) __NOEXC {
  return __sycl_std::__invoke_rint<half2>(x);
}
inline half3 rint(half3 x) __NOEXC {
  return __sycl_std::__invoke_rint<half3>(x);
}
inline half4 rint(half4 x) __NOEXC {
  return __sycl_std::__invoke_rint<half4>(x);
}
inline half8 rint(half8 x) __NOEXC {
  return __sycl_std::__invoke_rint<half8>(x);
}
inline half16 rint(half16 x) __NOEXC {
  return __sycl_std::__invoke_rint<half16>(x);
}

// genfloat rootn (genfloat x, genint y)
inline float rootn(float x, int y) __NOEXC {
  return __sycl_std::__invoke_rootn<float>(x, y);
}
inline float2 rootn(float2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_rootn<float2>(x, y);
}
inline float3 rootn(float3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_rootn<float3>(x, y);
}
inline float4 rootn(float4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_rootn<float4>(x, y);
}
inline float8 rootn(float8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_rootn<float8>(x, y);
}
inline float16 rootn(float16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_rootn<float16>(x, y);
}
inline double rootn(double x, int y) __NOEXC {
  return __sycl_std::__invoke_rootn<double>(x, y);
}
inline double2 rootn(double2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_rootn<double2>(x, y);
}
inline double3 rootn(double3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_rootn<double3>(x, y);
}
inline double4 rootn(double4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_rootn<double4>(x, y);
}
inline double8 rootn(double8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_rootn<double8>(x, y);
}
inline double16 rootn(double16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_rootn<double16>(x, y);
}
inline half rootn(half x, int y) __NOEXC {
  return __sycl_std::__invoke_rootn<half>(x, y);
}
inline half2 rootn(half2 x, int2 y) __NOEXC {
  return __sycl_std::__invoke_rootn<half2>(x, y);
}
inline half3 rootn(half3 x, int3 y) __NOEXC {
  return __sycl_std::__invoke_rootn<half3>(x, y);
}
inline half4 rootn(half4 x, int4 y) __NOEXC {
  return __sycl_std::__invoke_rootn<half4>(x, y);
}
inline half8 rootn(half8 x, int8 y) __NOEXC {
  return __sycl_std::__invoke_rootn<half8>(x, y);
}
inline half16 rootn(half16 x, int16 y) __NOEXC {
  return __sycl_std::__invoke_rootn<half16>(x, y);
}

// genfloat round (genfloat x)
inline float round(float x) __NOEXC {
  return __sycl_std::__invoke_round<float>(x);
}
inline float2 round(float2 x) __NOEXC {
  return __sycl_std::__invoke_round<float2>(x);
}
inline float3 round(float3 x) __NOEXC {
  return __sycl_std::__invoke_round<float3>(x);
}
inline float4 round(float4 x) __NOEXC {
  return __sycl_std::__invoke_round<float4>(x);
}
inline float8 round(float8 x) __NOEXC {
  return __sycl_std::__invoke_round<float8>(x);
}
inline float16 round(float16 x) __NOEXC {
  return __sycl_std::__invoke_round<float16>(x);
}
inline double round(double x) __NOEXC {
  return __sycl_std::__invoke_round<double>(x);
}
inline double2 round(double2 x) __NOEXC {
  return __sycl_std::__invoke_round<double2>(x);
}
inline double3 round(double3 x) __NOEXC {
  return __sycl_std::__invoke_round<double3>(x);
}
inline double4 round(double4 x) __NOEXC {
  return __sycl_std::__invoke_round<double4>(x);
}
inline double8 round(double8 x) __NOEXC {
  return __sycl_std::__invoke_round<double8>(x);
}
inline double16 round(double16 x) __NOEXC {
  return __sycl_std::__invoke_round<double16>(x);
}
inline half round(half x) __NOEXC {
  return __sycl_std::__invoke_round<half>(x);
}
inline half2 round(half2 x) __NOEXC {
  return __sycl_std::__invoke_round<half2>(x);
}
inline half3 round(half3 x) __NOEXC {
  return __sycl_std::__invoke_round<half3>(x);
}
inline half4 round(half4 x) __NOEXC {
  return __sycl_std::__invoke_round<half4>(x);
}
inline half8 round(half8 x) __NOEXC {
  return __sycl_std::__invoke_round<half8>(x);
}
inline half16 round(half16 x) __NOEXC {
  return __sycl_std::__invoke_round<half16>(x);
}

// genfloat rsqrt (genfloat x)
inline float rsqrt(float x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float>(x);
}
inline float2 rsqrt(float2 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float2>(x);
}
inline float3 rsqrt(float3 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float3>(x);
}
inline float4 rsqrt(float4 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float4>(x);
}
inline float8 rsqrt(float8 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float8>(x);
}
inline float16 rsqrt(float16 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<float16>(x);
}
inline double rsqrt(double x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double>(x);
}
inline double2 rsqrt(double2 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double2>(x);
}
inline double3 rsqrt(double3 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double3>(x);
}
inline double4 rsqrt(double4 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double4>(x);
}
inline double8 rsqrt(double8 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double8>(x);
}
inline double16 rsqrt(double16 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<double16>(x);
}
inline half rsqrt(half x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half>(x);
}
inline half2 rsqrt(half2 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half2>(x);
}
inline half3 rsqrt(half3 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half3>(x);
}
inline half4 rsqrt(half4 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half4>(x);
}
inline half8 rsqrt(half8 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half8>(x);
}
inline half16 rsqrt(half16 x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<half16>(x);
}

// genfloat sin (genfloat x)
inline float sin(float x) __NOEXC { return __sycl_std::__invoke_sin<float>(x); }
inline float2 sin(float2 x) __NOEXC {
  return __sycl_std::__invoke_sin<float2>(x);
}
inline float3 sin(float3 x) __NOEXC {
  return __sycl_std::__invoke_sin<float3>(x);
}
inline float4 sin(float4 x) __NOEXC {
  return __sycl_std::__invoke_sin<float4>(x);
}
inline float8 sin(float8 x) __NOEXC {
  return __sycl_std::__invoke_sin<float8>(x);
}
inline float16 sin(float16 x) __NOEXC {
  return __sycl_std::__invoke_sin<float16>(x);
}
inline double sin(double x) __NOEXC {
  return __sycl_std::__invoke_sin<double>(x);
}
inline double2 sin(double2 x) __NOEXC {
  return __sycl_std::__invoke_sin<double2>(x);
}
inline double3 sin(double3 x) __NOEXC {
  return __sycl_std::__invoke_sin<double3>(x);
}
inline double4 sin(double4 x) __NOEXC {
  return __sycl_std::__invoke_sin<double4>(x);
}
inline double8 sin(double8 x) __NOEXC {
  return __sycl_std::__invoke_sin<double8>(x);
}
inline double16 sin(double16 x) __NOEXC {
  return __sycl_std::__invoke_sin<double16>(x);
}
inline half sin(half x) __NOEXC { return __sycl_std::__invoke_sin<half>(x); }
inline half2 sin(half2 x) __NOEXC { return __sycl_std::__invoke_sin<half2>(x); }
inline half3 sin(half3 x) __NOEXC { return __sycl_std::__invoke_sin<half3>(x); }
inline half4 sin(half4 x) __NOEXC { return __sycl_std::__invoke_sin<half4>(x); }
inline half8 sin(half8 x) __NOEXC { return __sycl_std::__invoke_sin<half8>(x); }
inline half16 sin(half16 x) __NOEXC {
  return __sycl_std::__invoke_sin<half16>(x);
}

// genfloat sincos (genfloat x, genfloatptr cosval)
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float
sincos(float x, multi_ptr<float, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float2
sincos(float2 x, multi_ptr<float2, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float2>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float3
sincos(float3 x, multi_ptr<float3, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float3>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float4
sincos(float4 x, multi_ptr<float4, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float4>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float8
sincos(float8 x, multi_ptr<float8, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float8>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline float16
sincos(float16 x,
       multi_ptr<float16, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<float16>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double
sincos(double x, multi_ptr<double, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double2
sincos(double2 x,
       multi_ptr<double2, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double2>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double3
sincos(double3 x,
       multi_ptr<double3, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double3>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double4
sincos(double4 x,
       multi_ptr<double4, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double4>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double8
sincos(double8 x,
       multi_ptr<double8, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double8>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline double16
sincos(double16 x,
       multi_ptr<double16, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<double16>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half sincos(half x,
                   multi_ptr<half, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half2
sincos(half2 x, multi_ptr<half2, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half2>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half3
sincos(half3 x, multi_ptr<half3, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half3>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half4
sincos(half4 x, multi_ptr<half4, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half4>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half8
sincos(half8 x, multi_ptr<half8, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half8>(x, cosval);
}
template <access::address_space AddressSpace, access::decorated IsDecorated>
inline half16
sincos(half16 x, multi_ptr<half16, AddressSpace, IsDecorated> cosval) __NOEXC {
  return __sycl_std::__invoke_sincos<half16>(x, cosval);
}

// genfloat sinh (genfloat x)
inline float sinh(float x) __NOEXC {
  return __sycl_std::__invoke_sinh<float>(x);
}
inline float2 sinh(float2 x) __NOEXC {
  return __sycl_std::__invoke_sinh<float2>(x);
}
inline float3 sinh(float3 x) __NOEXC {
  return __sycl_std::__invoke_sinh<float3>(x);
}
inline float4 sinh(float4 x) __NOEXC {
  return __sycl_std::__invoke_sinh<float4>(x);
}
inline float8 sinh(float8 x) __NOEXC {
  return __sycl_std::__invoke_sinh<float8>(x);
}
inline float16 sinh(float16 x) __NOEXC {
  return __sycl_std::__invoke_sinh<float16>(x);
}
inline double sinh(double x) __NOEXC {
  return __sycl_std::__invoke_sinh<double>(x);
}
inline double2 sinh(double2 x) __NOEXC {
  return __sycl_std::__invoke_sinh<double2>(x);
}
inline double3 sinh(double3 x) __NOEXC {
  return __sycl_std::__invoke_sinh<double3>(x);
}
inline double4 sinh(double4 x) __NOEXC {
  return __sycl_std::__invoke_sinh<double4>(x);
}
inline double8 sinh(double8 x) __NOEXC {
  return __sycl_std::__invoke_sinh<double8>(x);
}
inline double16 sinh(double16 x) __NOEXC {
  return __sycl_std::__invoke_sinh<double16>(x);
}
inline half sinh(half x) __NOEXC { return __sycl_std::__invoke_sinh<half>(x); }
inline half2 sinh(half2 x) __NOEXC {
  return __sycl_std::__invoke_sinh<half2>(x);
}
inline half3 sinh(half3 x) __NOEXC {
  return __sycl_std::__invoke_sinh<half3>(x);
}
inline half4 sinh(half4 x) __NOEXC {
  return __sycl_std::__invoke_sinh<half4>(x);
}
inline half8 sinh(half8 x) __NOEXC {
  return __sycl_std::__invoke_sinh<half8>(x);
}
inline half16 sinh(half16 x) __NOEXC {
  return __sycl_std::__invoke_sinh<half16>(x);
}

// genfloat sinpi (genfloat x)
inline float sinpi(float x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float>(x);
}
inline float2 sinpi(float2 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float2>(x);
}
inline float3 sinpi(float3 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float3>(x);
}
inline float4 sinpi(float4 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float4>(x);
}
inline float8 sinpi(float8 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float8>(x);
}
inline float16 sinpi(float16 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<float16>(x);
}
inline double sinpi(double x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double>(x);
}
inline double2 sinpi(double2 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double2>(x);
}
inline double3 sinpi(double3 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double3>(x);
}
inline double4 sinpi(double4 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double4>(x);
}
inline double8 sinpi(double8 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double8>(x);
}
inline double16 sinpi(double16 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<double16>(x);
}
inline half sinpi(half x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half>(x);
}
inline half2 sinpi(half2 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half2>(x);
}
inline half3 sinpi(half3 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half3>(x);
}
inline half4 sinpi(half4 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half4>(x);
}
inline half8 sinpi(half8 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half8>(x);
}
inline half16 sinpi(half16 x) __NOEXC {
  return __sycl_std::__invoke_sinpi<half16>(x);
}

// genfloat sqrt (genfloat x)
inline float sqrt(float x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float>(x);
}
inline float2 sqrt(float2 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float2>(x);
}
inline float3 sqrt(float3 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float3>(x);
}
inline float4 sqrt(float4 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float4>(x);
}
inline float8 sqrt(float8 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float8>(x);
}
inline float16 sqrt(float16 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<float16>(x);
}
inline double sqrt(double x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double>(x);
}
inline double2 sqrt(double2 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double2>(x);
}
inline double3 sqrt(double3 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double3>(x);
}
inline double4 sqrt(double4 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double4>(x);
}
inline double8 sqrt(double8 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double8>(x);
}
inline double16 sqrt(double16 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<double16>(x);
}
inline half sqrt(half x) __NOEXC { return __sycl_std::__invoke_sqrt<half>(x); }
inline half2 sqrt(half2 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<half2>(x);
}
inline half3 sqrt(half3 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<half3>(x);
}
inline half4 sqrt(half4 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<half4>(x);
}
inline half8 sqrt(half8 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<half8>(x);
}
inline half16 sqrt(half16 x) __NOEXC {
  return __sycl_std::__invoke_sqrt<half16>(x);
}

// genfloat tan (genfloat x)
inline float tan(float x) __NOEXC { return __sycl_std::__invoke_tan<float>(x); }
inline float2 tan(float2 x) __NOEXC {
  return __sycl_std::__invoke_tan<float2>(x);
}
inline float3 tan(float3 x) __NOEXC {
  return __sycl_std::__invoke_tan<float3>(x);
}
inline float4 tan(float4 x) __NOEXC {
  return __sycl_std::__invoke_tan<float4>(x);
}
inline float8 tan(float8 x) __NOEXC {
  return __sycl_std::__invoke_tan<float8>(x);
}
inline float16 tan(float16 x) __NOEXC {
  return __sycl_std::__invoke_tan<float16>(x);
}
inline double tan(double x) __NOEXC {
  return __sycl_std::__invoke_tan<double>(x);
}
inline double2 tan(double2 x) __NOEXC {
  return __sycl_std::__invoke_tan<double2>(x);
}
inline double3 tan(double3 x) __NOEXC {
  return __sycl_std::__invoke_tan<double3>(x);
}
inline double4 tan(double4 x) __NOEXC {
  return __sycl_std::__invoke_tan<double4>(x);
}
inline double8 tan(double8 x) __NOEXC {
  return __sycl_std::__invoke_tan<double8>(x);
}
inline double16 tan(double16 x) __NOEXC {
  return __sycl_std::__invoke_tan<double16>(x);
}
inline half tan(half x) __NOEXC { return __sycl_std::__invoke_tan<half>(x); }
inline half2 tan(half2 x) __NOEXC { return __sycl_std::__invoke_tan<half2>(x); }
inline half3 tan(half3 x) __NOEXC { return __sycl_std::__invoke_tan<half3>(x); }
inline half4 tan(half4 x) __NOEXC { return __sycl_std::__invoke_tan<half4>(x); }
inline half8 tan(half8 x) __NOEXC { return __sycl_std::__invoke_tan<half8>(x); }
inline half16 tan(half16 x) __NOEXC {
  return __sycl_std::__invoke_tan<half16>(x);
}

// genfloat tanh (genfloat x)
inline float tanh(float x) __NOEXC {
  return __sycl_std::__invoke_tanh<float>(x);
}
inline float2 tanh(float2 x) __NOEXC {
  return __sycl_std::__invoke_tanh<float2>(x);
}
inline float3 tanh(float3 x) __NOEXC {
  return __sycl_std::__invoke_tanh<float3>(x);
}
inline float4 tanh(float4 x) __NOEXC {
  return __sycl_std::__invoke_tanh<float4>(x);
}
inline float8 tanh(float8 x) __NOEXC {
  return __sycl_std::__invoke_tanh<float8>(x);
}
inline float16 tanh(float16 x) __NOEXC {
  return __sycl_std::__invoke_tanh<float16>(x);
}
inline double tanh(double x) __NOEXC {
  return __sycl_std::__invoke_tanh<double>(x);
}
inline double2 tanh(double2 x) __NOEXC {
  return __sycl_std::__invoke_tanh<double2>(x);
}
inline double3 tanh(double3 x) __NOEXC {
  return __sycl_std::__invoke_tanh<double3>(x);
}
inline double4 tanh(double4 x) __NOEXC {
  return __sycl_std::__invoke_tanh<double4>(x);
}
inline double8 tanh(double8 x) __NOEXC {
  return __sycl_std::__invoke_tanh<double8>(x);
}
inline double16 tanh(double16 x) __NOEXC {
  return __sycl_std::__invoke_tanh<double16>(x);
}
inline half tanh(half x) __NOEXC { return __sycl_std::__invoke_tanh<half>(x); }
inline half2 tanh(half2 x) __NOEXC {
  return __sycl_std::__invoke_tanh<half2>(x);
}
inline half3 tanh(half3 x) __NOEXC {
  return __sycl_std::__invoke_tanh<half3>(x);
}
inline half4 tanh(half4 x) __NOEXC {
  return __sycl_std::__invoke_tanh<half4>(x);
}
inline half8 tanh(half8 x) __NOEXC {
  return __sycl_std::__invoke_tanh<half8>(x);
}
inline half16 tanh(half16 x) __NOEXC {
  return __sycl_std::__invoke_tanh<half16>(x);
}

// genfloat tanpi (genfloat x)
inline float tanpi(float x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float>(x);
}
inline float2 tanpi(float2 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float2>(x);
}
inline float3 tanpi(float3 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float3>(x);
}
inline float4 tanpi(float4 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float4>(x);
}
inline float8 tanpi(float8 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float8>(x);
}
inline float16 tanpi(float16 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<float16>(x);
}
inline double tanpi(double x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double>(x);
}
inline double2 tanpi(double2 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double2>(x);
}
inline double3 tanpi(double3 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double3>(x);
}
inline double4 tanpi(double4 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double4>(x);
}
inline double8 tanpi(double8 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double8>(x);
}
inline double16 tanpi(double16 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<double16>(x);
}
inline half tanpi(half x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half>(x);
}
inline half2 tanpi(half2 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half2>(x);
}
inline half3 tanpi(half3 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half3>(x);
}
inline half4 tanpi(half4 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half4>(x);
}
inline half8 tanpi(half8 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half8>(x);
}
inline half16 tanpi(half16 x) __NOEXC {
  return __sycl_std::__invoke_tanpi<half16>(x);
}

// genfloat tgamma (genfloat x)
inline float tgamma(float x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float>(x);
}
inline float2 tgamma(float2 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float2>(x);
}
inline float3 tgamma(float3 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float3>(x);
}
inline float4 tgamma(float4 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float4>(x);
}
inline float8 tgamma(float8 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float8>(x);
}
inline float16 tgamma(float16 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<float16>(x);
}
inline double tgamma(double x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double>(x);
}
inline double2 tgamma(double2 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double2>(x);
}
inline double3 tgamma(double3 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double3>(x);
}
inline double4 tgamma(double4 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double4>(x);
}
inline double8 tgamma(double8 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double8>(x);
}
inline double16 tgamma(double16 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<double16>(x);
}
inline half tgamma(half x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half>(x);
}
inline half2 tgamma(half2 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half2>(x);
}
inline half3 tgamma(half3 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half3>(x);
}
inline half4 tgamma(half4 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half4>(x);
}
inline half8 tgamma(half8 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half8>(x);
}
inline half16 tgamma(half16 x) __NOEXC {
  return __sycl_std::__invoke_tgamma<half16>(x);
}

// genfloat trunc (genfloat x)
inline float trunc(float x) __NOEXC {
  return __sycl_std::__invoke_trunc<float>(x);
}
inline float2 trunc(float2 x) __NOEXC {
  return __sycl_std::__invoke_trunc<float2>(x);
}
inline float3 trunc(float3 x) __NOEXC {
  return __sycl_std::__invoke_trunc<float3>(x);
}
inline float4 trunc(float4 x) __NOEXC {
  return __sycl_std::__invoke_trunc<float4>(x);
}
inline float8 trunc(float8 x) __NOEXC {
  return __sycl_std::__invoke_trunc<float8>(x);
}
inline float16 trunc(float16 x) __NOEXC {
  return __sycl_std::__invoke_trunc<float16>(x);
}
inline double trunc(double x) __NOEXC {
  return __sycl_std::__invoke_trunc<double>(x);
}
inline double2 trunc(double2 x) __NOEXC {
  return __sycl_std::__invoke_trunc<double2>(x);
}
inline double3 trunc(double3 x) __NOEXC {
  return __sycl_std::__invoke_trunc<double3>(x);
}
inline double4 trunc(double4 x) __NOEXC {
  return __sycl_std::__invoke_trunc<double4>(x);
}
inline double8 trunc(double8 x) __NOEXC {
  return __sycl_std::__invoke_trunc<double8>(x);
}
inline double16 trunc(double16 x) __NOEXC {
  return __sycl_std::__invoke_trunc<double16>(x);
}
inline half trunc(half x) __NOEXC {
  return __sycl_std::__invoke_trunc<half>(x);
}
inline half2 trunc(half2 x) __NOEXC {
  return __sycl_std::__invoke_trunc<half2>(x);
}
inline half3 trunc(half3 x) __NOEXC {
  return __sycl_std::__invoke_trunc<half3>(x);
}
inline half4 trunc(half4 x) __NOEXC {
  return __sycl_std::__invoke_trunc<half4>(x);
}
inline half8 trunc(half8 x) __NOEXC {
  return __sycl_std::__invoke_trunc<half8>(x);
}
inline half16 trunc(half16 x) __NOEXC {
  return __sycl_std::__invoke_trunc<half16>(x);
}

// other marray math functions

// TODO: can be optimized in the way marray math functions above are optimized
// (usage of vec<T, 2>)
#define __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARGPTR,   \
                                                               ...)            \
  marray<T, N> res;                                                            \
  for (int j = 0; j < N; j++) {                                                \
    res[j] =                                                                   \
        NAME(__VA_ARGS__,                                                      \
             address_space_cast<AddressSpace, IsDecorated,                     \
                                detail::marray_element_t<T2>>(&(*ARGPTR)[j])); \
  }                                                                            \
  return res;

#define __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genfloatptr_marray<T2, AddressSpace, IsDecorated>::value, \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2)       \
      __NOEXC {                                                                \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARG2,         \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(fract, x, iptr,
                                                               x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(modf, x, iptr,
                                                               x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENFLOATPTR_OVERLOAD(sincos, x,
                                                               cosval, x[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_GENFLOATPTR_OVERLOAD

#define __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(          \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genintptr_marray<T2, AddressSpace, IsDecorated>::value,   \
      marray<T, N>>                                                            \
  NAME(marray<T, N> ARG1, multi_ptr<T2, AddressSpace, IsDecorated> ARG2)       \
      __NOEXC {                                                                \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, ARG2,         \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(frexp, x, exp,
                                                             x[j])
__SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_ARG_GENINTPTR_OVERLOAD(lgamma_r, x, signp,
                                                             x[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_BINOP_2ND_GENINTPTR_OVERLOAD

#define __SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD(NAME, ...)                 \
  template <typename T, size_t N, typename T2,                                 \
            access::address_space AddressSpace, access::decorated IsDecorated> \
  std::enable_if_t<                                                            \
      detail::is_svgenfloat<T>::value &&                                       \
          detail::is_genintptr_marray<T2, AddressSpace, IsDecorated>::value,   \
      marray<T, N>>                                                            \
  NAME(marray<T, N> x, marray<T, N> y,                                         \
       multi_ptr<T2, AddressSpace, IsDecorated> quo) __NOEXC {                 \
    __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL(NAME, quo,          \
                                                           __VA_ARGS__)        \
  }

__SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD(remquo, x[j], y[j])

#undef __SYCL_MARRAY_MATH_FUNCTION_REMQUO_OVERLOAD

#undef __SYCL_MARRAY_MATH_FUNCTION_W_GENPTR_ARG_OVERLOAD_IMPL

template <typename T, size_t N>
std::enable_if_t<detail::is_nan_type<T>::value,
                 marray<detail::nan_return_t<T>, N>>
nan(marray<T, N> nancode) __NOEXC {
  marray<detail::nan_return_t<T>, N> res;
  for (int j = 0; j < N; j++) {
    res[j] = nan(nancode[j]);
  }
  return res;
}

/* --------------- 4.13.5 Common functions. ---------------------------------*/
// svgenfloat clamp (svgenfloat x, svgenfloat minval, svgenfloat maxval)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> clamp(T x, T minval,
                                                           T maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, minval, maxval);
}

// vgenfloath clamp (vgenfloath x, half minval, half maxval)
// vgenfloatf clamp (vgenfloatf x, float minval, float maxval)
// vgenfloatd clamp (vgenfloatd x, double minval, double maxval)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_fclamp<T>(x, T(minval), T(maxval));
}

// svgenfloat degrees (svgenfloat radians)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
degrees(T radians) __NOEXC {
  return __sycl_std::__invoke_degrees<T>(radians);
}

// svgenfloat abs (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_fabs<T>(x);
}

// svgenfloat max (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, y);
}

// vgenfloatf max (vgenfloatf x, float y)
// vgenfloatd max (vgenfloatd x, double y)
// vgenfloath max (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmax_common<T>(x, T(y));
}

// svgenfloat min (svgenfloat x, svgenfloat y)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, y);
}

// vgenfloatf min (vgenfloatf x, float y)
// vgenfloatd min (vgenfloatd x, double y)
// vgenfloath min (vgenfloath x, half y)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_fmin_common<T>(x, T(y));
}

// svgenfloat mix (svgenfloat x, svgenfloat y, svgenfloat a)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> mix(T x, T y,
                                                         T a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, a);
}

// vgenfloatf mix (vgenfloatf x, vgenfloatf y, float a)
// vgenfloatd mix (vgenfloatd x, vgenfloatd y, double a)
// vgenfloatd mix (vgenfloath x, vgenfloath y, half a)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
mix(T x, T y, typename T::element_type a) __NOEXC {
  return __sycl_std::__invoke_mix<T>(x, y, T(a));
}

// svgenfloat radians (svgenfloat degrees)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
radians(T degrees) __NOEXC {
  return __sycl_std::__invoke_radians<T>(degrees);
}

// svgenfloat step (svgenfloat edge, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> step(T edge, T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(edge, x);
}

// vgenfloatf step (float edge, vgenfloatf x)
// vgenfloatd step (double edge, vgenfloatd x)
// vgenfloatd step (half edge, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
step(typename T::element_type edge, T x) __NOEXC {
  return __sycl_std::__invoke_step<T>(T(edge), x);
}

// svgenfloat smoothstep (svgenfloat edge0, svgenfloat edge1, svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T>
smoothstep(T edge0, T edge1, T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(edge0, edge1, x);
}

// vgenfloatf smoothstep (float edge0, float edge1, vgenfloatf x)
// vgenfloatd smoothstep (double edge0, double edge1, vgenfloatd x)
// vgenfloath smoothstep (half edge0, half edge1, vgenfloath x)
template <typename T>
std::enable_if_t<detail::is_vgenfloat<T>::value, T>
smoothstep(typename T::element_type edge0, typename T::element_type edge1,
           T x) __NOEXC {
  return __sycl_std::__invoke_smoothstep<T>(T(edge0), T(edge1), x);
}

// svgenfloat sign (svgenfloat x)
template <typename T>
std::enable_if_t<detail::is_svgenfloat<T>::value, T> sign(T x) __NOEXC {
  return __sycl_std::__invoke_sign<T>(x);
}

// marray common functions

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, ...)                 \
  T res;                                                                       \
  for (int i = 0; i < T::size(); i++) {                                        \
    res[i] = NAME(__VA_ARGS__);                                                \
  }                                                                            \
  return res;

#define __SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(NAME, ARG, ...)            \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG) __NOEXC {                                                        \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(degrees, T radians, radians[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(radians, T degrees, degrees[i])
__SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD(sign, T x, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(NAME, ARG1, ARG2, ...)    \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG1, ARG2) __NOEXC {                                                 \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

// min and max may be defined as macros, so we wrap them in parentheses to avoid
// errors.
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((min), T x, T y, x[i], y[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((min), T x,
                                             detail::marray_element_t<T> y,
                                             x[i], y)
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((max), T x, T y, x[i], y[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD((max), T x,
                                             detail::marray_element_t<T> y,
                                             x[i], y)
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(step, T edge, T x, edge[i], x[i])
__SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD(step,
                                             detail::marray_element_t<T> edge,
                                             T x, edge, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_BINOP_OVERLOAD

#define __SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(NAME, ARG1, ARG2, ARG3,   \
                                                     ...)                      \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  T NAME(ARG1, ARG2, ARG3) __NOEXC {                                           \
    __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)             \
  }

__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(clamp, T x, T minval, T maxval,
                                             x[i], minval[i], maxval[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(
    clamp, T x, detail::marray_element_t<T> minval,
    detail::marray_element_t<T> maxval, x[i], minval, maxval)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y, T a, x[i], y[i],
                                             a[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(mix, T x, T y,
                                             detail::marray_element_t<T> a,
                                             x[i], y[i], a)
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(smoothstep, T edge0, T edge1, T x,
                                             edge0[i], edge1[i], x[i])
__SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD(
    smoothstep, detail::marray_element_t<T> edge0,
    detail::marray_element_t<T> edge1, T x, edge0, edge1, x[i])

#undef __SYCL_MARRAY_COMMON_FUNCTION_TEROP_OVERLOAD
#undef __SYCL_MARRAY_COMMON_FUNCTION_OVERLOAD_IMPL

/* --------------- 4.13.4 Integer functions. --------------------------------*/
// ugeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> abs(T x) __NOEXC {
  return __sycl_std::__invoke_u_abs<T>(x);
}

// igeninteger abs (geninteger x)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> abs(T x) __NOEXC {
  auto res = __sycl_std::__invoke_s_abs<detail::make_unsigned_t<T>>(x);
  if constexpr (detail::is_vigeninteger<T>::value) {
    return res.template convert<detail::vector_element_t<T>>();
  } else
    return detail::make_signed_t<decltype(res)>(res);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> abs_diff(T x,
                                                               T y) __NOEXC {
  return __sycl_std::__invoke_u_abs_diff<T>(x, y);
}

// ugeninteger abs_diff (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, detail::make_unsigned_t<T>>
abs_diff(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_abs_diff<detail::make_unsigned_t<T>>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> add_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_add_sat<T>(x, y);
}

// geninteger add_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> add_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_add_sat<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> hadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_hadd<T>(x, y);
}

// geninteger hadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> hadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_hadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> rhadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_rhadd<T>(x, y);
}

// geninteger rhadd (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> rhadd(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_rhadd<T>(x, y);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> clamp(T x, T minval,
                                                            T maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, geninteger minval, geninteger maxval)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> clamp(T x, T minval,
                                                            T maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, minval, maxval);
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_s_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clamp (geninteger x, sgeninteger minval, sgeninteger maxval)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>
clamp(T x, typename T::element_type minval,
      typename T::element_type maxval) __NOEXC {
  return __sycl_std::__invoke_u_clamp<T>(x, T(minval), T(maxval));
}

// geninteger clz (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> clz(T x) __NOEXC {
  return __sycl_std::__invoke_clz<T>(x);
}

// geninteger ctz (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> ctz(T x) __NOEXC {
  return __sycl_std::__invoke_ctz<T>(x);
}

// geninteger ctz (geninteger x) for calls with deprecated namespace
namespace ext::intel {
template <typename T>
__SYCL_DEPRECATED(
    "'sycl::ext::intel::ctz' is deprecated, use 'sycl::ctz' instead")
std::enable_if_t<sycl::detail::is_geninteger<T>::value, T> ctz(T x) __NOEXC {
  return sycl::ctz(x);
}
} // namespace ext::intel

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") intel {
using namespace ext::intel;
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mad_hi(T x, T y,
                                                             T z) __NOEXC {
  return __sycl_std::__invoke_s_mad_hi<T>(x, y, z);
}

// geninteger mad_hi (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_hi(T x, T y,
                                                             T z) __NOEXC {
  return __sycl_std::__invoke_u_mad_hi<T>(x, y, z);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mad_sat(T a, T b,
                                                              T c) __NOEXC {
  return __sycl_std::__invoke_s_mad_sat<T>(a, b, c);
}

// geninteger mad_sat (geninteger a, geninteger b, geninteger c)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mad_sat(T a, T b,
                                                              T c) __NOEXC {
  return __sycl_std::__invoke_u_mad_sat<T>(a, b, c);
}

// igeninteger max (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, y);
}

// ugeninteger max (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T>(max)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, y);
}

// igeninteger max (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_max<T>(x, T(y));
}

// vugeninteger max (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>(max)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_max<T>(x, T(y));
}

// igeninteger min (igeninteger x, igeninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, y);
}

// ugeninteger min (ugeninteger x, ugeninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T>(min)(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, y);
}

// vigeninteger min (vigeninteger x, sigeninteger y)
template <typename T>
std::enable_if_t<detail::is_vigeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_s_min<T>(x, T(y));
}

// vugeninteger min (vugeninteger x, sugeninteger y)
template <typename T>
std::enable_if_t<detail::is_vugeninteger<T>::value, T>(min)(
    T x, typename T::element_type y) __NOEXC {
  return __sycl_std::__invoke_u_min<T>(x, T(y));
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> mul_hi(T x, T y) __NOEXC {
  return __sycl_std::__invoke_s_mul_hi<T>(x, y);
}

// geninteger mul_hi (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> mul_hi(T x, T y) __NOEXC {
  return __sycl_std::__invoke_u_mul_hi<T>(x, y);
}

// geninteger rotate (geninteger v, geninteger i)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> rotate(T v, T i) __NOEXC {
  return __sycl_std::__invoke_rotate<T>(v, i);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_igeninteger<T>::value, T> sub_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_s_sub_sat<T>(x, y);
}

// geninteger sub_sat (geninteger x, geninteger y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger<T>::value, T> sub_sat(T x,
                                                              T y) __NOEXC {
  return __sycl_std::__invoke_u_sub_sat<T>(x, y);
}

// ugeninteger16bit upsample (ugeninteger8bit hi, ugeninteger8bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger8bit<T>::value, detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger16bit upsample (igeninteger8bit hi, ugeninteger8bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger8bit<T>::value &&
                     detail::is_ugeninteger8bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger32bit upsample (ugeninteger16bit hi, ugeninteger16bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger16bit<T>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger32bit upsample (igeninteger16bit hi, ugeninteger16bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger16bit<T>::value &&
                     detail::is_ugeninteger16bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// ugeninteger64bit upsample (ugeninteger32bit hi, ugeninteger32bit lo)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T lo) __NOEXC {
  return __sycl_std::__invoke_u_upsample<detail::make_larger_t<T>>(hi, lo);
}

// igeninteger64bit upsample (igeninteger32bit hi, ugeninteger32bit lo)
template <typename T, typename T2>
std::enable_if_t<detail::is_igeninteger32bit<T>::value &&
                     detail::is_ugeninteger32bit<T2>::value,
                 detail::make_larger_t<T>>
upsample(T hi, T2 lo) __NOEXC {
  detail::check_vector_size<T, T2>();
  return __sycl_std::__invoke_s_upsample<detail::make_larger_t<T>>(hi, lo);
}

// geninteger popcount (geninteger x)
template <typename T>
std::enable_if_t<detail::is_geninteger<T>::value, T> popcount(T x) __NOEXC {
  return __sycl_std::__invoke_popcount<T>(x);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit<T>::value, T> mad24(T x, T y,
                                                                 T z) __NOEXC {
  return __sycl_std::__invoke_s_mad24<T>(x, y, z);
}

// geninteger32bit mad24 (geninteger32bit x, geninteger32bit y,
// geninteger32bit z)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value, T> mad24(T x, T y,
                                                                 T z) __NOEXC {
  return __sycl_std::__invoke_u_mad24<T>(x, y, z);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_igeninteger32bit<T>::value, T> mul24(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_s_mul24<T>(x, y);
}

// geninteger32bit mul24 (geninteger32bit x, geninteger32bit y)
template <typename T>
std::enable_if_t<detail::is_ugeninteger32bit<T>::value, T> mul24(T x,
                                                                 T y) __NOEXC {
  return __sycl_std::__invoke_u_mul24<T>(x, y);
}

// marray integer functions

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, ...)                \
  marray<T, N> res;                                                            \
  for (int j = 0; j < N; j++) {                                                \
    res[j] = NAME(__VA_ARGS__);                                                \
  }                                                                            \
  return res;

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD(NAME, ARG, ...)          \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(NAME, ARG, ...)          \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD(abs, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD(abs, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_ABS_U_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(NAME, ARG, ...)           \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_geninteger<T>::value, marray<T, N>> NAME(        \
      marray<T, N> ARG) __NOEXC {                                              \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(clz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(ctz, x, x[j])
__SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD(popcount, x, x[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_UNOP_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD(NAME, ARG1,      \
                                                              ARG2, ...)       \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value,                           \
                   marray<detail::make_unsigned_t<T>, N>>                      \
  NAME(marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                         \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2) __NOEXC {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD(        \
    NAME, ARG1, ARG2, ...)                                                     \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2) __NOEXC {                                     \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(abs_diff, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD(abs_diff, x, y, x[j],
                                                      y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(add_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(add_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(hadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(hadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(rhadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(rhadd, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD((max), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD((max), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD((max), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD((max), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD((min), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD((min), x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD((min), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD((min), x, y,
                                                               x[j], y)
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(mul_hi, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(mul_hi, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(rotate, v, i, v[j], i[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(rotate, v, i, v[j], i[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD(sub_sat, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD(sub_sat, x, y, x[j], y[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_2ND_ARG_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_2ND_ARG_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_I_RET_U_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_BINOP_U_OVERLOAD

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2, T ARG3) __NOEXC {                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD(   \
    NAME, ARG1, ARG2, ARG3, ...)                                               \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger<T>::value, marray<T, N>> NAME(       \
      marray<T, N> ARG1, T ARG2, T ARG3) __NOEXC {                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(clamp, x, minval, maxval, x[j],
                                                minval[j], maxval[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(clamp, x, minval, maxval, x[j],
                                                minval[j], maxval[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD(
    clamp, x, minval, maxval, x[j], minval, maxval)
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD(
    clamp, x, minval, maxval, x[j], minval, maxval)
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(mad_hi, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(mad_hi, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD(mad_sat, a, b, c, a[j], b[j],
                                                c[j])
__SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD(mad_sat, a, b, c, a[j], b[j],
                                                c[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_2ND_3RD_ARGS_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_2ND_3RD_ARGS_SCALAR_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_TEROP_U_OVERLOAD

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD(NAME, ARG1, ARG2,      \
                                                        ARG3, ...)             \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2, marray<T, N> ARG3) __NOEXC {       \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD(mad24, x, y, z, x[j], y[j],
                                                z[j])
__SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD(mad24, x, y, z, x[j], y[j],
                                                z[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_MAD24_U_OVERLOAD

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD(NAME, ARG1, ARG2, ...) \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_igeninteger32bit<T>::value, marray<T, N>> NAME(  \
      marray<T, N> ARG1, marray<T, N> ARG2) __NOEXC {                          \
    __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL(NAME, __VA_ARGS__)            \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD(mul24, x, y, x[j], y[j])
__SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD(mul24, x, y, x[j], y[j])

#undef __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_I_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_MUL24_U_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_OVERLOAD_IMPL

// TODO: can be optimized in the way math functions are optimized (usage of
// vec<T, 2>)
#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)            \
  detail::make_larger_t<marray<T, N>> res;                                     \
  for (int j = 0; j < N; j++) {                                                \
    res[j] = NAME(hi[j], lo[j]);                                               \
  }                                                                            \
  return res;

// Keep NAME for readability
#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(NAME, KBIT)        \
  template <typename T, size_t N>                                              \
  std::enable_if_t<detail::is_ugeninteger##KBIT<T>::value,                     \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T, N> lo) __NOEXC {                             \
    __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)                \
  }

#define __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(NAME, KBIT)        \
  template <typename T, typename T2, size_t N>                                 \
  std::enable_if_t<detail::is_igeninteger##KBIT<T>::value &&                   \
                       detail::is_ugeninteger##KBIT<T2>::value,                \
                   detail::make_larger_t<marray<T, N>>>                        \
  NAME(marray<T, N> hi, marray<T2, N> lo) __NOEXC {                            \
    __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL(NAME)                \
  }

__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 8bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 8bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 16bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 16bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD(upsample, 32bit)
__SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD(upsample, 32bit)

#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_IU_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_UU_OVERLOAD
#undef __SYCL_MARRAY_INTEGER_FUNCTION_UPSAMPLE_OVERLOAD_IMPL

/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
// float3 cross (float3 p0, float3 p1)
// float4 cross (float4 p0, float4 p1)
// double3 cross (double3 p0, double3 p1)
// double4 cross (double4 p0, double4 p1)
// half3 cross (half3 p0, half3 p1)
// half4 cross (half4 p0, half4 p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE cross(TYPE p0, TYPE p1) __NOEXC {                                \
    return __sycl_std::__invoke_cross<TYPE>(p0, p1);                           \
  }
__SYCL_DEF_BUILTIN_VGENGEOCROSSFLOAT
#undef __SYCL_BUILTIN_DEF
#undef __SYCL_DEF_BUILTIN_VGENGEOCROSSFLOAT
#undef __SYCL_DEF_BUILTIN_HALF_GEOCROSSVEC
#undef __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSVEC
#undef __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSVEC
#undef __SYCL_DEF_BUILTIN_GEOCROSSVEC

// float dot (float p0, float p1)
// double dot (double p0, double p1)
// half dot (half p0, half p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE dot(TYPE p0, TYPE p1) __NOEXC { return p0 * p1; }
__SYCL_DEF_BUILTIN_SGENFLOAT
#undef __SYCL_BUILTIN_DEF
// float dot (vgengeofloat p0, vgengeofloat p1)
// double dot (vgengeodouble p0, vgengeodouble p1)
// half dot (vgengeohalf p0, vgengeohalf p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::element_type dot(TYPE p0, TYPE p1) __NOEXC {                    \
    return __sycl_std::__invoke_Dot<TYPE::element_type>(p0, p1);               \
  }
__SYCL_DEF_BUILTIN_VGENGEOFLOAT
#undef __SYCL_BUILTIN_DEF

// float distance (float p0, float p1)
// double distance (double p0, double p1)
// half distance (half p0, half p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE distance(TYPE p0, TYPE p1) __NOEXC {                             \
    return __sycl_std::__invoke_distance<TYPE>(p0, p1);                        \
  }
__SYCL_DEF_BUILTIN_SGENFLOAT
#undef __SYCL_BUILTIN_DEF
// float distance (vgengeofloat p0, vgengeofloat p1)
// double distance (vgengeodouble p0, vgengeodouble p1)
// half distance (vgengeohalf p0, vgengeohalf p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::element_type distance(TYPE p0, TYPE p1) __NOEXC {               \
    return __sycl_std::__invoke_distance<TYPE::element_type>(p0, p1);          \
  }
__SYCL_DEF_BUILTIN_VGENGEOFLOAT
#undef __SYCL_BUILTIN_DEF

// float length (float p0, float p1)
// double length (double p0, double p1)
// half length (half p0, half p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE length(TYPE p) __NOEXC {                                         \
    return __sycl_std::__invoke_length<TYPE>(p);                               \
  }
__SYCL_DEF_BUILTIN_SGENFLOAT
#undef __SYCL_BUILTIN_DEF
// float length (vgengeofloat p0, vgengeofloat p1)
// double length (vgengeodouble p0, vgengeodouble p1)
// half length (vgengeohalf p0, vgengeohalf p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::element_type length(TYPE p) __NOEXC {                           \
    return __sycl_std::__invoke_length<TYPE::element_type>(p);                 \
  }
__SYCL_DEF_BUILTIN_VGENGEOFLOAT
#undef __SYCL_BUILTIN_DEF

// gengeofloat normalize (gengeofloat p)
// gengeodouble normalize (gengeodouble p)
// gengeohalf normalize (gengeohalf p)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE normalize(TYPE p) __NOEXC {                                      \
    return __sycl_std::__invoke_normalize<TYPE>(p);                            \
  }
__SYCL_DEF_BUILTIN_GENGEOFLOAT
#undef __SYCL_BUILTIN_DEF

// float fast_distance (gengeofloat p0, gengeofloat p1)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline float fast_distance(TYPE p0, TYPE p1) __NOEXC {                       \
    return __sycl_std::__invoke_fast_distance<float>(p0, p1);                  \
  }
__SYCL_DEF_BUILTIN_GENGEOFLOATF
#undef __SYCL_BUILTIN_DEF

// float fast_length (gengeofloat p)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline float fast_length(TYPE p) __NOEXC {                                   \
    return __sycl_std::__invoke_fast_length<float>(p);                         \
  }
__SYCL_DEF_BUILTIN_GENGEOFLOATF
#undef __SYCL_BUILTIN_DEF

// gengeofloat fast_normalize (gengeofloat p)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE fast_normalize(TYPE p) __NOEXC {                                 \
    return __sycl_std::__invoke_fast_normalize<TYPE>(p);                       \
  }
__SYCL_DEF_BUILTIN_GENGEOFLOATF
#undef __SYCL_BUILTIN_DEF

// marray geometric functions

// cross
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE cross(TYPE p0, TYPE p1) __NOEXC {                                \
    return detail::to_marray(cross(detail::to_vec(p0), detail::to_vec(p1)));   \
  }
__SYCL_DEF_BUILTIN_GENGEOCROSSMARRAY
#undef __SYCL_BUILTIN_DEF

#undef __SYCL_DEF_BUILTIN_GENGEOCROSSMARRAY
#undef __SYCL_DEF_BUILTIN_HALF_GEOCROSSMARRAY
#undef __SYCL_DEF_BUILTIN_DOUBLE_GEOCROSSMARRAY
#undef __SYCL_DEF_BUILTIN_FLOAT_GEOCROSSMARRAY
#undef __SYCL_DEF_BUILTIN_GEOCROSSMARRAY

// dot
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::value_type dot(TYPE p0, TYPE p1) __NOEXC {                      \
    return dot(detail::to_vec(p0), detail::to_vec(p1));                        \
  }
__SYCL_DEF_BUILTIN_GENGEOMARRAY
#undef __SYCL_BUILTIN_DEF

// distance
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::value_type distance(TYPE p0, TYPE p1) __NOEXC {                 \
    return distance(detail::to_vec(p0), detail::to_vec(p1));                   \
  }
__SYCL_DEF_BUILTIN_GENGEOMARRAY
#undef __SYCL_BUILTIN_DEF

// length
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE::value_type length(TYPE p) __NOEXC {                             \
    return length(detail::to_vec(p));                                          \
  }
__SYCL_DEF_BUILTIN_GENGEOMARRAY
#undef __SYCL_BUILTIN_DEF

// normalize
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE normalize(TYPE p) __NOEXC {                                      \
    return detail::to_marray(normalize(detail::to_vec(p)));                    \
  }
__SYCL_DEF_BUILTIN_GENGEOMARRAY
#undef __SYCL_BUILTIN_DEF

// fast_distance
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline float fast_distance(TYPE p0, TYPE p1) __NOEXC {                       \
    return fast_distance(detail::to_vec(p0), detail::to_vec(p1));              \
  }
__SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY
#undef __SYCL_BUILTIN_DEF

// fast_normalize
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE fast_normalize(TYPE p) __NOEXC {                                 \
    return detail::to_marray(fast_normalize(detail::to_vec(p)));               \
  }
__SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY
#undef __SYCL_BUILTIN_DEF

// fast_length
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline float fast_length(TYPE p) __NOEXC {                                   \
    return fast_length(detail::to_vec(p));                                     \
  }
__SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY
#undef __SYCL_BUILTIN_DEF

#undef __SYCL_DEF_BUILTIN_GENGEOMARRAY
#undef __SYCL_DEF_BUILTIN_HALF_GEOMARRAY
#undef __SYCL_DEF_BUILTIN_DOUBLE_GEOMARRAY
#undef __SYCL_DEF_BUILTIN_FLOAT_GEOMARRAY
#undef __SYCL_DEF_BUILTIN_GEOMARRAY

/* SYCL 1.2.1 ---- 4.13.7 Relational functions. -----------------------------*/
/* SYCL 2020  ---- 4.17.9 Relational functions. -----------------------------*/

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isequal(TYPE x, TYPE y) __NOEXC {      \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdEqual<detail::internal_rel_ret_t<TYPE>>(x,    \
                                                                         y));  \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isnotequal(TYPE x, TYPE y) __NOEXC {   \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FUnordNotEqual<detail::internal_rel_ret_t<TYPE>>( \
            x, y));                                                            \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isgreater(TYPE x, TYPE y) __NOEXC {    \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdGreaterThan<                                  \
            detail::internal_rel_ret_t<TYPE>>(x, y));                          \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isgreaterequal(TYPE x, TYPE y)         \
      __NOEXC {                                                                \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdGreaterThanEqual<                             \
            detail::internal_rel_ret_t<TYPE>>(x, y));                          \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isless(TYPE x, TYPE y) __NOEXC {       \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdLessThan<detail::internal_rel_ret_t<TYPE>>(   \
            x, y));                                                            \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> islessequal(TYPE x, TYPE y) __NOEXC {  \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdLessThanEqual<                                \
            detail::internal_rel_ret_t<TYPE>>(x, y));                          \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> islessgreater(TYPE x, TYPE y)          \
      __NOEXC {                                                                \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_FOrdNotEqual<detail::internal_rel_ret_t<TYPE>>(   \
            x, y));                                                            \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isfinite(TYPE x) __NOEXC {             \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_IsFinite<detail::internal_rel_ret_t<TYPE>>(x));   \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isinf(TYPE x) __NOEXC {                \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_IsInf<detail::internal_rel_ret_t<TYPE>>(x));      \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isnan(TYPE x) __NOEXC {                \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_IsNan<detail::internal_rel_ret_t<TYPE>>(x));      \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isnormal(TYPE x) __NOEXC {             \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_IsNormal<detail::internal_rel_ret_t<TYPE>>(x));   \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isordered(TYPE x, TYPE y) __NOEXC {    \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_Ordered<detail::internal_rel_ret_t<TYPE>>(x, y)); \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> isunordered(TYPE x, TYPE y) __NOEXC {  \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_Unordered<detail::internal_rel_ret_t<TYPE>>(x,    \
                                                                         y));  \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::common_rel_ret_t<TYPE> signbit(TYPE x) __NOEXC {              \
    return detail::RelConverter<TYPE>::apply(                                  \
        __sycl_std::__invoke_SignBitSet<detail::internal_rel_ret_t<TYPE>>(x)); \
  }
__SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_BUILTIN_DEF

// marray relational functions

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(NAME)                 \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  sycl::marray<bool, T::size()> NAME(T x, T y) __NOEXC {                       \
    sycl::marray<bool, T::size()> res;                                         \
    for (int i = 0; i < x.size(); i++) {                                       \
      res[i] = NAME(x[i], y[i]);                                               \
    }                                                                          \
    return res;                                                                \
  }

#define __SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(NAME)                  \
  template <typename T,                                                        \
            typename = std::enable_if_t<detail::is_mgenfloat<T>::value>>       \
  sycl::marray<bool, T::size()> NAME(T x) __NOEXC {                            \
    sycl::marray<bool, T::size()> res;                                         \
    for (int i = 0; i < x.size(); i++) {                                       \
      res[i] = NAME(x[i]);                                                     \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isnotequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isgreater)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isgreaterequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isless)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(islessequal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(islessgreater)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isfinite)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isinf)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isnan)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(isnormal)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isordered)
__SYCL_MARRAY_RELATIONAL_FUNCTION_BINOP_OVERLOAD(isunordered)
__SYCL_MARRAY_RELATIONAL_FUNCTION_UNOP_OVERLOAD(signbit)

// bool any (sigeninteger x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline bool any(TYPE x) __NOEXC {                                            \
    return detail::Boolean<1>(int(detail::msbIsSet(x)));                       \
  }
__SYCL_DEF_BUILTIN_SIGENINTEGER
#undef __SYCL_BUILTIN_DEF

// int any (vigeninteger x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline int any(TYPE x) __NOEXC {                                             \
    return detail::rel_sign_bit_test_ret_t<TYPE>(                              \
        __sycl_std::__invoke_Any<detail::rel_sign_bit_test_ret_t<TYPE>>(       \
            detail::rel_sign_bit_test_arg_t<TYPE>(x)));                        \
  }
__SYCL_DEF_BUILTIN_VIGENINTEGER
#undef __SYCL_BUILTIN_DEF

// bool all (sigeninteger x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline bool all(TYPE x) __NOEXC {                                            \
    return detail::Boolean<1>(int(detail::msbIsSet(x)));                       \
  }
__SYCL_DEF_BUILTIN_SIGENINTEGER
#undef __SYCL_BUILTIN_DEF

// int all (vigeninteger x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline int all(TYPE x) __NOEXC {                                             \
    return detail::rel_sign_bit_test_ret_t<TYPE>(                              \
        __sycl_std::__invoke_All<detail::rel_sign_bit_test_ret_t<TYPE>>(       \
            detail::rel_sign_bit_test_arg_t<TYPE>(x)));                        \
  }
__SYCL_DEF_BUILTIN_VIGENINTEGER
#undef __SYCL_BUILTIN_DEF

// gentype bitselect (gentype a, gentype b, gentype c)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE bitselect(TYPE a, TYPE b, TYPE c) __NOEXC {                      \
    return __sycl_std::__invoke_bitselect<TYPE>(a, b, c);                      \
  }
__SYCL_DEF_BUILTIN_GENTYPE
#undef __SYCL_BUILTIN_DEF

// sgentype select (sgentype a, sgentype b, bool c)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE select(TYPE a, TYPE b, bool c) __NOEXC {                         \
    /* sycl::select(sgentype a, sgentype b, bool c) calls OpenCL built-in      \
     select(sgentype a, sgentype b, igentype c). This type trait makes the     \
     proper conversion for argument c from bool to igentype, based on sgentype \
     == T.*/                                                                   \
    using get_select_opencl_builtin_c_arg_type =                               \
        detail::same_size_int_t<TYPE, std::is_signed_v<TYPE>>;                 \
                                                                               \
    return __sycl_std::__invoke_select<TYPE>(                                  \
        a, b, static_cast<get_select_opencl_builtin_c_arg_type>(c));           \
  }
__SYCL_DEF_BUILTIN_SGENTYPE
#undef __SYCL_BUILTIN_DEF

// vgentype select(vgentype a, vgentype b, vigeninteger c)
// vgentype select(vgentype a, vgentype b, vugeninteger c)
// Non-standard:
// sgentype select(sgentype a, sgentype b, sigeninteger c)
// sgentype select(sgentype a, sgentype b, sugeninteger c)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE select(TYPE a, TYPE b, detail::same_size_int_t<TYPE, true> c)    \
      __NOEXC {                                                                \
    return __sycl_std::__invoke_select<TYPE>(a, b, c);                         \
  }                                                                            \
  inline TYPE select(TYPE a, TYPE b, detail::same_size_int_t<TYPE, false> c)   \
      __NOEXC {                                                                \
    return __sycl_std::__invoke_select<TYPE>(a, b, c);                         \
  }
__SYCL_DEF_BUILTIN_VGENTYPE
__SYCL_DEF_BUILTIN_SGENTYPE
#undef __SYCL_BUILTIN_DEF

// Since same_size_int_t uses long long for 64-bit as it is guaranteed to have
// the appropriate size, we need special cases for long.
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline detail::same_size_int_t<TYPE, true> select(                           \
      detail::same_size_int_t<TYPE, true> a,                                   \
      detail::same_size_int_t<TYPE, true> b, TYPE c) __NOEXC {                 \
    return __sycl_std::__invoke_select<detail::same_size_int_t<TYPE, true>>(   \
        a, b, c);                                                              \
  }                                                                            \
  inline detail::same_size_int_t<TYPE, false> select(                          \
      detail::same_size_int_t<TYPE, false> a,                                  \
      detail::same_size_int_t<TYPE, false> b, TYPE c) __NOEXC {                \
    return __sycl_std::__invoke_select<detail::same_size_int_t<TYPE, false>>(  \
        a, b, c);                                                              \
  }                                                                            \
  inline detail::same_size_float_t<TYPE> select(                               \
      detail::same_size_float_t<TYPE> a, detail::same_size_float_t<TYPE> b,    \
      TYPE c) __NOEXC {                                                        \
    return __sycl_std::__invoke_select<detail::same_size_float_t<TYPE>>(a, b,  \
                                                                        c);    \
  }
__SYCL_DEF_BUILTIN_LONG_SCALAR
__SYCL_DEF_BUILTIN_ULONG_SCALAR
#undef __SYCL_BUILTIN_DEF

// other marray relational functions

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool>
any(marray<T, N> x) __NOEXC {
  return std::any_of(x.begin(), x.end(), [](T i) { return any(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_sigeninteger<T>::value, bool>
all(marray<T, N> x) __NOEXC {
  return std::all_of(x.begin(), x.end(), [](T i) { return all(i); });
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype<T>::value, marray<T, N>>
bitselect(marray<T, N> a, marray<T, N> b, marray<T, N> c) __NOEXC {
  marray<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = bitselect(a[i], b[i], c[i]);
  }
  return res;
}

template <typename T, size_t N>
std::enable_if_t<detail::is_gentype<T>::value, marray<T, N>>
select(marray<T, N> a, marray<T, N> b, marray<bool, N> c) __NOEXC {
  marray<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = select(a[i], b[i], c[i]);
  }
  return res;
}

namespace native {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/

#define __SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(NAME)                             \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x)        \
      __NOEXC {                                                                \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_native_##NAME<vec<float, 2>>(    \
          detail::to_vec2(x, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_native_##NAME<float>(x[N - 1]);        \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(sin)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(cos)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(tan)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp2)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(exp10)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log2)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(log10)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(sqrt)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(rsqrt)
__SYCL_NATIVE_MATH_FUNCTION_OVERLOAD(recip)

#undef __SYCL_NATIVE_MATH_FUNCTION_OVERLOAD

#define __SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(NAME)                           \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(                           \
      marray<float, N> x, marray<float, N> y) __NOEXC {                        \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_native_##NAME<vec<float, 2>>(    \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));               \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_native_##NAME<float>(x[N - 1], y[N - 1]);       \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(divide)
__SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD(powr)

#undef __SYCL_NATIVE_MATH_FUNCTION_2_OVERLOAD

// genfloatf cos (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE cos(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_native_cos<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf divide (genfloatf x, genfloatf y)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE divide(TYPE x, TYPE y) __NOEXC {                                 \
    return __sycl_std::__invoke_native_divide<TYPE>(x, y);                     \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_native_exp<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp2(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_native_exp2<TYPE>(x);                          \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp10(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_native_exp10<TYPE>(x);                         \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_native_log<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log2(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_native_log2<TYPE>(x);                          \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log10(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_native_log10<TYPE>(x);                         \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf powr (genfloatf x, genfloatf y)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE powr(TYPE x, TYPE y) __NOEXC {                                   \
    return __sycl_std::__invoke_native_powr<TYPE>(x, y);                       \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf recip (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE recip(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_native_recip<TYPE>(x);                         \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf rsqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE rsqrt(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_native_rsqrt<TYPE>(x);                         \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sin (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sin(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_native_sin<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sqrt(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_native_sqrt<TYPE>(x);                          \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf tan (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE tan(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_native_tan<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

} // namespace native
namespace half_precision {
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
#define __SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(NAME)                     \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(marray<float, N> x)        \
      __NOEXC {                                                                \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_half_##NAME<vec<float, 2>>(      \
          detail::to_vec2(x, i * 2));                                          \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] = __sycl_std::__invoke_half_##NAME<float>(x[N - 1]);          \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(sin)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(cos)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(tan)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp2)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(exp10)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log2)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(log10)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(sqrt)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(rsqrt)
__SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD(recip)

#undef __SYCL_HALF_PRECISION_MATH_FUNCTION_OVERLOAD

#define __SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(NAME)                   \
  template <size_t N>                                                          \
  inline __SYCL_ALWAYS_INLINE marray<float, N> NAME(                           \
      marray<float, N> x, marray<float, N> y) __NOEXC {                        \
    marray<float, N> res;                                                      \
    for (size_t i = 0; i < N / 2; i++) {                                       \
      auto partial_res = __sycl_std::__invoke_half_##NAME<vec<float, 2>>(      \
          detail::to_vec2(x, i * 2), detail::to_vec2(y, i * 2));               \
      std::memcpy(&res[i * 2], &partial_res, sizeof(vec<float, 2>));           \
    }                                                                          \
    if (N % 2) {                                                               \
      res[N - 1] =                                                             \
          __sycl_std::__invoke_half_##NAME<float>(x[N - 1], y[N - 1]);         \
    }                                                                          \
    return res;                                                                \
  }

__SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(divide)
__SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD(powr)

#undef __SYCL_HALF_PRECISION_MATH_FUNCTION_2_OVERLOAD

// genfloatf cos (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE cos(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_half_cos<TYPE>(x);                             \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf divide (genfloatf x, genfloatf y)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE divide(TYPE x, TYPE y) __NOEXC {                                 \
    return __sycl_std::__invoke_half_divide<TYPE>(x, y);                       \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_half_exp<TYPE>(x);                             \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp2(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_half_exp2<TYPE>(x);                            \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp10(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_half_exp10<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_half_log<TYPE>(x);                             \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log2(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_half_log2<TYPE>(x);                            \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log10(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_half_log10<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf powr (genfloatf x, genfloatf y)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE powr(TYPE x, TYPE y) __NOEXC {                                   \
    return __sycl_std::__invoke_half_powr<TYPE>(x, y);                         \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf recip (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE recip(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_half_recip<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf rsqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE rsqrt(TYPE x) __NOEXC {                                          \
    return __sycl_std::__invoke_half_rsqrt<TYPE>(x);                           \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sin (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sin(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_half_sin<TYPE>(x);                             \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sqrt(TYPE x) __NOEXC {                                           \
    return __sycl_std::__invoke_half_sqrt<TYPE>(x);                            \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf tan (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE tan(TYPE x) __NOEXC {                                            \
    return __sycl_std::__invoke_half_tan<TYPE>(x);                             \
  }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

} // namespace half_precision

#ifdef __FAST_MATH__
/* ----------------- -ffast-math functions. ---------------------------------*/

#define __SYCL_MATH_FUNCTION_OVERLOAD_FM(NAME)                                 \
  template <typename T, size_t N>                                              \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<std::is_same_v<T, float>, marray<T, N>>                 \
      NAME(marray<T, N> x) __NOEXC {                                           \
    return native::NAME(x);                                                    \
  }

__SYCL_MATH_FUNCTION_OVERLOAD_FM(sin)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(cos)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(tan)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(exp10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log2)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(log10)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(sqrt)
__SYCL_MATH_FUNCTION_OVERLOAD_FM(rsqrt)
#undef __SYCL_MATH_FUNCTION_OVERLOAD_FM

// genfloatf cos (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE cos(TYPE x) __NOEXC { return native::cos(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp(TYPE x) __NOEXC { return native::exp(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp2(TYPE x) __NOEXC { return native::exp2(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf exp10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE exp10(TYPE x) __NOEXC { return native::exp10(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log(genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log(TYPE x) __NOEXC { return native::log(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log2 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log2(TYPE x) __NOEXC { return native::log2(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf log10 (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE log10(TYPE x) __NOEXC { return native::log10(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf powr (genfloatf x, genfloatf y)
// TODO: remove when __SYCL_DEF_BUILTIN_MARRAY is defined
template <typename T, size_t N>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<std::is_same_v<T, float>, marray<T, N>>
    powr(marray<T, N> x, marray<T, N> y) __NOEXC {
  return native::powr(x, y);
}

#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE powr(TYPE x, TYPE y) __NOEXC { return native::powr(x, y); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf rsqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE rsqrt(TYPE x) __NOEXC { return native::rsqrt(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sin (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sin(TYPE x) __NOEXC { return native::sin(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf sqrt (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE sqrt(TYPE x) __NOEXC { return native::sqrt(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

// genfloatf tan (genfloatf x)
#define __SYCL_BUILTIN_DEF(TYPE)                                               \
  inline TYPE tan(TYPE x) __NOEXC { return native::tan(x); }
__SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_BUILTIN_DEF

#endif // __FAST_MATH__

#undef __SYCL_DEF_BUILTIN_VEC
#undef __SYCL_DEF_BUILTIN_GEOVEC
#undef __SYCL_DEF_BUILTIN_MARRAY
#undef __SYCL_DEF_BUILTIN_CHAR_SCALAR
#undef __SYCL_DEF_BUILTIN_CHAR_VEC
#undef __SYCL_DEF_BUILTIN_CHAR_MARRAY
#undef __SYCL_DEF_BUILTIN_CHARN
#undef __SYCL_DEF_BUILTIN_SCHAR_SCALAR
#undef __SYCL_DEF_BUILTIN_SCHAR_VEC
#undef __SYCL_DEF_BUILTIN_SCHAR_MARRAY
#undef __SYCL_DEF_BUILTIN_SCHARN
#undef __SYCL_DEF_BUILTIN_IGENCHAR
#undef __SYCL_DEF_BUILTIN_UCHAR_SCALAR
#undef __SYCL_DEF_BUILTIN_UCHAR_VEC
#undef __SYCL_DEF_BUILTIN_UCHAR_MARRAY
#undef __SYCL_DEF_BUILTIN_UCHARN
#undef __SYCL_DEF_BUILTIN_UGENCHAR
#undef __SYCL_DEF_BUILTIN_GENCHAR
#undef __SYCL_DEF_BUILTIN_SHORT_SCALAR
#undef __SYCL_DEF_BUILTIN_SHORT_VEC
#undef __SYCL_DEF_BUILTIN_SHORT_MARRAY
#undef __SYCL_DEF_BUILTIN_SHORTN
#undef __SYCL_DEF_BUILTIN_GENSHORT
#undef __SYCL_DEF_BUILTIN_USHORT_SCALAR
#undef __SYCL_DEF_BUILTIN_USHORT_MARRAY
#undef __SYCL_DEF_BUILTIN_USHORTN
#undef __SYCL_DEF_BUILTIN_UGENSHORT
#undef __SYCL_DEF_BUILTIN_INT_SCALAR
#undef __SYCL_DEF_BUILTIN_INT_VEC
#undef __SYCL_DEF_BUILTIN_INT_MARRAY
#undef __SYCL_DEF_BUILTIN_INTN
#undef __SYCL_DEF_BUILTIN_GENINT
#undef __SYCL_DEF_BUILTIN_UINT_SCALAR
#undef __SYCL_DEF_BUILTIN_UINT_VEC
#undef __SYCL_DEF_BUILTIN_UINT_MARRAY
#undef __SYCL_DEF_BUILTIN_UINTN
#undef __SYCL_DEF_BUILTIN_UGENINT
#undef __SYCL_DEF_BUILTIN_LONG_SCALAR
#undef __SYCL_DEF_BUILTIN_LONG_VEC
#undef __SYCL_DEF_BUILTIN_LONG_MARRAY
#undef __SYCL_DEF_BUILTIN_LONGN
#undef __SYCL_DEF_BUILTIN_GENLONG
#undef __SYCL_DEF_BUILTIN_ULONG_SCALAR
#undef __SYCL_DEF_BUILTIN_ULONG_VEC
#undef __SYCL_DEF_BUILTIN_ULONG_MARRAY
#undef __SYCL_DEF_BUILTIN_ULONGN
#undef __SYCL_DEF_BUILTIN_UGENLONG
#undef __SYCL_DEF_BUILTIN_LONGLONG_SCALAR
#undef __SYCL_DEF_BUILTIN_LONGLONG_VEC
#undef __SYCL_DEF_BUILTIN_LONGLONG_MARRAY
#undef __SYCL_DEF_BUILTIN_LONGLONGN
#undef __SYCL_DEF_BUILTIN_GENLONGLONG
#undef __SYCL_DEF_BUILTIN_ULONGLONG_SCALAR
#undef __SYCL_DEF_BUILTIN_ULONGLONG_VEC
#undef __SYCL_DEF_BUILTIN_ULONGLONG_MARRAY
#undef __SYCL_DEF_BUILTIN_ULONGLONGN
#undef __SYCL_DEF_BUILTIN_UGENLONGLONG
#undef __SYCL_DEF_BUILTIN_IGENLONGINTEGER
#undef __SYCL_DEF_BUILTIN_UGENLONGINTEGER
#undef __SYCL_DEF_BUILTIN_SIGENINTEGER
#undef __SYCL_DEF_BUILTIN_VIGENINTEGER
#undef __SYCL_DEF_BUILTIN_IGENINTEGER
#undef __SYCL_DEF_BUILTIN_SUGENINTEGER
#undef __SYCL_DEF_BUILTIN_VUGENINTEGER
#undef __SYCL_DEF_BUILTIN_UGENINTEGER
#undef __SYCL_DEF_BUILTIN_SGENINTEGER
#undef __SYCL_DEF_BUILTIN_VGENINTEGER
#undef __SYCL_DEF_BUILTIN_GENINTEGER
#undef __SYCL_DEF_BUILTIN_FLOAT_SCALAR
#undef __SYCL_DEF_BUILTIN_FLOAT_VEC
#undef __SYCL_DEF_BUILTIN_FLOAT_GEOVEC
#undef __SYCL_DEF_BUILTIN_FLOAT_MARRAY
#undef __SYCL_DEF_BUILTIN_FLOATN
#undef __SYCL_DEF_BUILTIN_GENFLOATF
#undef __SYCL_DEF_BUILTIN_GENGEOFLOATF
#undef __SYCL_DEF_BUILTIN_DOUBLE_SCALAR
#undef __SYCL_DEF_BUILTIN_DOUBLE_VEC
#undef __SYCL_DEF_BUILTIN_DOUBLE_GEOVEC
#undef __SYCL_DEF_BUILTIN_DOUBLE_MARRAY
#undef __SYCL_DEF_BUILTIN_DOUBLEN
#undef __SYCL_DEF_BUILTIN_GENFLOATD
#undef __SYCL_DEF_BUILTIN_GENGEOFLOATD
#undef __SYCL_DEF_BUILTIN_HALF_SCALAR
#undef __SYCL_DEF_BUILTIN_HALF_VEC
#undef __SYCL_DEF_BUILTIN_HALF_GEOVEC
#undef __SYCL_DEF_BUILTIN_HALF_MARRAY
#undef __SYCL_DEF_BUILTIN_HALFN
#undef __SYCL_DEF_BUILTIN_GENFLOATH
#undef __SYCL_DEF_BUILTIN_GENGEOFLOATH
#undef __SYCL_DEF_BUILTIN_SGENFLOAT
#undef __SYCL_DEF_BUILTIN_VGENFLOAT
#undef __SYCL_DEF_BUILTIN_GENFLOAT
#undef __SYCL_DEF_BUILTIN_GENGEOFLOAT
#undef __SYCL_DEF_BUILTIN_FAST_MATH_GENFLOAT
#undef __SYCL_DEF_BUILTIN_SGENTYPE
#undef __SYCL_DEF_BUILTIN_VGENTYPE
#undef __SYCL_DEF_BUILTIN_GENTYPE
#undef __SYCL_COMMA
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace
  // sycl

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {
extern __DPCPP_SYCL_EXTERNAL int abs(int x);
extern __DPCPP_SYCL_EXTERNAL long int labs(long int x);
extern __DPCPP_SYCL_EXTERNAL long long int llabs(long long int x);

extern __DPCPP_SYCL_EXTERNAL div_t div(int x, int y);
extern __DPCPP_SYCL_EXTERNAL ldiv_t ldiv(long int x, long int y);
extern __DPCPP_SYCL_EXTERNAL lldiv_t lldiv(long long int x, long long int y);
extern __DPCPP_SYCL_EXTERNAL float scalbnf(float x, int n);
extern __DPCPP_SYCL_EXTERNAL double scalbn(double x, int n);
extern __DPCPP_SYCL_EXTERNAL float logf(float x);
extern __DPCPP_SYCL_EXTERNAL double log(double x);
extern __DPCPP_SYCL_EXTERNAL float expf(float x);
extern __DPCPP_SYCL_EXTERNAL double exp(double x);
extern __DPCPP_SYCL_EXTERNAL float log10f(float x);
extern __DPCPP_SYCL_EXTERNAL double log10(double x);
extern __DPCPP_SYCL_EXTERNAL float modff(float x, float *intpart);
extern __DPCPP_SYCL_EXTERNAL double modf(double x, double *intpart);
extern __DPCPP_SYCL_EXTERNAL float exp2f(float x);
extern __DPCPP_SYCL_EXTERNAL double exp2(double x);
extern __DPCPP_SYCL_EXTERNAL float expm1f(float x);
extern __DPCPP_SYCL_EXTERNAL double expm1(double x);
extern __DPCPP_SYCL_EXTERNAL int ilogbf(float x);
extern __DPCPP_SYCL_EXTERNAL int ilogb(double x);
extern __DPCPP_SYCL_EXTERNAL float log1pf(float x);
extern __DPCPP_SYCL_EXTERNAL double log1p(double x);
extern __DPCPP_SYCL_EXTERNAL float log2f(float x);
extern __DPCPP_SYCL_EXTERNAL double log2(double x);
extern __DPCPP_SYCL_EXTERNAL float logbf(float x);
extern __DPCPP_SYCL_EXTERNAL double logb(double x);
extern __DPCPP_SYCL_EXTERNAL float sqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL double sqrt(double x);
extern __DPCPP_SYCL_EXTERNAL float cbrtf(float x);
extern __DPCPP_SYCL_EXTERNAL double cbrt(double x);
extern __DPCPP_SYCL_EXTERNAL float erff(float x);
extern __DPCPP_SYCL_EXTERNAL double erf(double x);
extern __DPCPP_SYCL_EXTERNAL float erfcf(float x);
extern __DPCPP_SYCL_EXTERNAL double erfc(double x);
extern __DPCPP_SYCL_EXTERNAL float tgammaf(float x);
extern __DPCPP_SYCL_EXTERNAL double tgamma(double x);
extern __DPCPP_SYCL_EXTERNAL float lgammaf(float x);
extern __DPCPP_SYCL_EXTERNAL double lgamma(double x);
extern __DPCPP_SYCL_EXTERNAL float fmodf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double fmod(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float remainderf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double remainder(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float remquof(float x, float y, int *q);
extern __DPCPP_SYCL_EXTERNAL double remquo(double x, double y, int *q);
extern __DPCPP_SYCL_EXTERNAL float nextafterf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double nextafter(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float fdimf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double fdim(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float fmaf(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL double fma(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL float sinf(float x);
extern __DPCPP_SYCL_EXTERNAL double sin(double x);
extern __DPCPP_SYCL_EXTERNAL float cosf(float x);
extern __DPCPP_SYCL_EXTERNAL double cos(double x);
extern __DPCPP_SYCL_EXTERNAL float tanf(float x);
extern __DPCPP_SYCL_EXTERNAL double tan(double x);
extern __DPCPP_SYCL_EXTERNAL float asinf(float x);
extern __DPCPP_SYCL_EXTERNAL double asin(double x);
extern __DPCPP_SYCL_EXTERNAL float acosf(float x);
extern __DPCPP_SYCL_EXTERNAL double acos(double x);
extern __DPCPP_SYCL_EXTERNAL float atanf(float x);
extern __DPCPP_SYCL_EXTERNAL double atan(double x);
extern __DPCPP_SYCL_EXTERNAL float powf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double pow(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float atan2f(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double atan2(double x, double y);

extern __DPCPP_SYCL_EXTERNAL float sinhf(float x);
extern __DPCPP_SYCL_EXTERNAL double sinh(double x);
extern __DPCPP_SYCL_EXTERNAL float coshf(float x);
extern __DPCPP_SYCL_EXTERNAL double cosh(double x);
extern __DPCPP_SYCL_EXTERNAL float tanhf(float x);
extern __DPCPP_SYCL_EXTERNAL double tanh(double x);
extern __DPCPP_SYCL_EXTERNAL float asinhf(float x);
extern __DPCPP_SYCL_EXTERNAL double asinh(double x);
extern __DPCPP_SYCL_EXTERNAL float acoshf(float x);
extern __DPCPP_SYCL_EXTERNAL double acosh(double x);
extern __DPCPP_SYCL_EXTERNAL float atanhf(float x);
extern __DPCPP_SYCL_EXTERNAL double atanh(double x);
extern __DPCPP_SYCL_EXTERNAL double frexp(double x, int *exp);
extern __DPCPP_SYCL_EXTERNAL double ldexp(double x, int exp);
extern __DPCPP_SYCL_EXTERNAL double hypot(double x, double y);

extern __DPCPP_SYCL_EXTERNAL void *memcpy(void *dest, const void *src,
                                          size_t n);
extern __DPCPP_SYCL_EXTERNAL void *memset(void *dest, int c, size_t n);
extern __DPCPP_SYCL_EXTERNAL int memcmp(const void *s1, const void *s2,
                                        size_t n);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmax(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmin(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmax(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmin(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umax(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umin(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_brev(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_brevll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_byte_perm(unsigned int x, unsigned int y, unsigned int s);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffs(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffsll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clz(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clzll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popc(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popcll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_sad(int x, int y,
                                                    unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_usad(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_rhadd(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_urhadd(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_uhadd(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mul24(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umul24(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mulhi(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umulhi(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_mul64hi(long long int x,
                                                         long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_umul64hi(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_abs(int x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llabs(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_saturatef(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fabsf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_floorf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ceilf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_truncf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_nearbyintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rsqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_invf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaxf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fminf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_copysignf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_exp10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_expf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_logf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log2f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_powf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_fdividef(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rd(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rn(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_ru(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rz(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rd(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rn(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_ru(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float_as_int(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float_as_uint(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rd(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rn(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_ru(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rz(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int_as_float(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint_as_float(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rd(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rn(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_ru(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half_as_short(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half_as_ushort(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rd(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rn(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_ru(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rz(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rd(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rn(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_ru(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rz(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short_as_half(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort_as_half(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_double2half(double x);

extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaf16(_Float16 x, _Float16 y,
                                                   _Float16 z);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fabsf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_floorf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ceilf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_truncf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_nearbyintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_sqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rsqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_invf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaxf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fminf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_copysignf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL float __imf_bfloat162float(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rd(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rn(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_ru(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rz(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rd(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rn(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_ru(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rz(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rd(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rn(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_ru(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rz(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_double2bfloat16(double x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat16_as_short(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat16_as_ushort(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short_as_bfloat16(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort_as_bfloat16(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmabf16(uint16_t x, uint16_t y,
                                                    uint16_t z);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmaxbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fminbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fabsbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rintbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_floorbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ceilbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_truncbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_copysignbf16(uint16_t x,
                                                         uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_sqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rsqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fabs(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_floor(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ceil(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_trunc(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_nearbyint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rsqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_inv(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmax(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmin(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_copysign(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rd(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rn(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_ru(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rz(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2hiint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2loint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rd(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rn(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_ru(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_int2double_rn(int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rd(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rn(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_ru(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double_as_longlong(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_longlong_as_double(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_hiloint2double(int hi, int lo);

extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu4(unsigned int x,
                                                       unsigned int y);
}
#ifdef __GLIBC__
extern "C" {
extern __DPCPP_SYCL_EXTERNAL void __assert_fail(const char *expr,
                                                const char *file,
                                                unsigned int line,
                                                const char *func);
extern __DPCPP_SYCL_EXTERNAL float frexpf(float x, int *exp);
extern __DPCPP_SYCL_EXTERNAL float ldexpf(float x, int exp);
extern __DPCPP_SYCL_EXTERNAL float hypotf(float x, float y);

// MS UCRT supports most of the C standard library but <complex.h> is
// an exception.
extern __DPCPP_SYCL_EXTERNAL float cimagf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double cimag(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float crealf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double creal(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float cargf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double carg(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float cabsf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double cabs(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cprojf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cproj(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cexpf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cexp(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ clogf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ clog(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cpowf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cpow(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csqrtf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csqrt(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csinhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csinh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ccoshf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ccosh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ctanhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ctanh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ csinf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ csin(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ccosf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ccos(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ ctanf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ ctan(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cacosf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cacos(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cacoshf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cacosh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ casinf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ casin(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ casinhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ casinh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ catanf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ catan(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ catanhf(float __complex__ z);
extern __DPCPP_SYCL_EXTERNAL double __complex__ catanh(double __complex__ z);
extern __DPCPP_SYCL_EXTERNAL float __complex__ cpolarf(float rho, float theta);
extern __DPCPP_SYCL_EXTERNAL double __complex__ cpolar(double rho,
                                                       double theta);
extern __DPCPP_SYCL_EXTERNAL float __complex__ __mulsc3(float a, float b,
                                                        float c, float d);
extern __DPCPP_SYCL_EXTERNAL double __complex__ __muldc3(double a, double b,
                                                         double c, double d);
extern __DPCPP_SYCL_EXTERNAL float __complex__ __divsc3(float a, float b,
                                                        float c, float d);
extern __DPCPP_SYCL_EXTERNAL double __complex__ __divdc3(float a, float b,
                                                         float c, float d);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern __DPCPP_SYCL_EXTERNAL double _Cosh(double x, double y);
extern __DPCPP_SYCL_EXTERNAL int _dpcomp(double x, double y);
extern __DPCPP_SYCL_EXTERNAL int _dsign(double x);
extern __DPCPP_SYCL_EXTERNAL short _Dtest(double *px);
extern __DPCPP_SYCL_EXTERNAL short _dtest(double *px);
extern __DPCPP_SYCL_EXTERNAL short _Exp(double *px, double y, short eoff);
extern __DPCPP_SYCL_EXTERNAL float _FCosh(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int _fdpcomp(float x, float y);
extern __DPCPP_SYCL_EXTERNAL int _fdsign(float x);
extern __DPCPP_SYCL_EXTERNAL short _FDtest(float *px);
extern __DPCPP_SYCL_EXTERNAL short _fdtest(float *px);
extern __DPCPP_SYCL_EXTERNAL short _FExp(float *px, float y, short eoff);
extern __DPCPP_SYCL_EXTERNAL float _FSinh(float x, float y);
extern __DPCPP_SYCL_EXTERNAL double _Sinh(double x, double y);
extern __DPCPP_SYCL_EXTERNAL float _hypotf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL void _wassert(const wchar_t *wexpr,
                                           const wchar_t *wfile, unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__

#undef __NOEXC
