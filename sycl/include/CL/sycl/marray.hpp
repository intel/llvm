//==----------------- marray.hpp --- Implements marray classes -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/aliases.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/half_type.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

/// Provides a cross-patform math array class template that works on
/// SYCL devices as well as in host C++ code.
///
/// \ingroup sycl_api
template <typename Type, std::size_t NumElements> class marray {
  using DataT = Type;

public:
  using value_type = Type;
  using reference = Type &;
  using const_reference = const Type &;
  using iterator = Type *;
  using const_iterator = const Type *;

private:
  value_type MData[NumElements];

  template <class...> struct conjunction : std::true_type {};
  template <class B1, class... tail>
  struct conjunction<B1, tail...>
      : std::conditional<bool(B1::value), conjunction<tail...>, B1>::type {};

  // TypeChecker is needed for (const ArgTN &... Args) ctor to validate Args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};

  // Shortcuts for Args validation in (const ArgTN &... Args) ctor.
  template <typename... ArgTN>
  using EnableIfSuitableTypes = typename std::enable_if<
      conjunction<TypeChecker<ArgTN, DataT>...>::value>::type;

public:
  constexpr marray() : MData{} {}

  explicit constexpr marray(const Type &Arg) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Arg;
    }
  }

  template <
      typename... ArgTN, typename = EnableIfSuitableTypes<ArgTN...>,
      typename = typename std::enable_if<sizeof...(ArgTN) == NumElements>::type>
  constexpr marray(const ArgTN &... Args) : MData{Args...} {}

  constexpr marray(const marray<Type, NumElements> &Rhs) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Rhs.MData[I];
    }
  }

  constexpr marray(marray<Type, NumElements> &&Rhs) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Rhs.MData[I];
    }
  }

  // Available only when: NumElements == 1
  template <std::size_t Size = NumElements,
            typename = typename std::enable_if<Size == 1>>
  operator Type() const {
    return MData[0];
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  // subscript operator
  reference operator[](std::size_t index) { return MData[index]; }

  const_reference operator[](std::size_t index) const { return MData[index]; }

  marray &operator=(const marray<Type, NumElements> &Rhs) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Rhs.MData[I];
    }
    return *this;
  }

  // broadcasting operator
  marray &operator=(const Type &Rhs) {
    for (std::size_t I = 0; I < NumElements; ++I) {
      MData[I] = Rhs;
    }
    return *this;
  }

  // iterator functions
  iterator begin() { return MData; }

  const_iterator begin() const { return MData; }

  iterator end() { return MData + NumElements; }

  const_iterator end() const { return MData + NumElements; }

#ifdef __SYCL_BINOP
#error "Undefine __SYCL_BINOP macro"
#endif

#ifdef __SYCL_BINOP_INTEGRAL
#error "Undefine __SYCL_BINOP_INTEGRAL macro"
#endif

#define __SYCL_BINOP(BINOP, OPASSIGN)                                          \
  friend marray operator BINOP(const marray &Lhs, const marray &Rhs) {         \
    marray Ret;                                                                \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if<                                              \
      std::is_convertible<DataT, T>::value &&                                  \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      marray>::type                                                            \
  operator BINOP(const marray &Lhs, const T &Rhs) {                            \
    return Lhs BINOP marray(static_cast<DataT>(Rhs));                          \
  }                                                                            \
  friend marray &operator OPASSIGN(marray &Lhs, const marray &Rhs) {           \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements>                                     \
  friend typename std::enable_if<Num != 1, marray &>::type operator OPASSIGN(  \
      marray &Lhs, const DataT &Rhs) {                                         \
    Lhs = Lhs BINOP marray(Rhs);                                               \
    return Lhs;                                                                \
  }

#define __SYCL_BINOP_INTEGRAL(BINOP, OPASSIGN)                                 \
  template <typename T = DataT>                                                \
  friend typename std::enable_if<std::is_integral<T>::value, marray>           \
  operator BINOP(const marray &Lhs, const marray &Rhs) {                       \
    marray Ret;                                                                \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     std::is_integral<T>::value &&             \
                                     std::is_integral<BaseT>::value,           \
                                 marray>::type                                 \
  operator BINOP(const marray &Lhs, const T &Rhs) {                            \
    return Lhs BINOP marray(static_cast<DataT>(Rhs));                          \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend typename std::enable_if<std::is_integral<T>::value, marray>           \
      &operator OPASSIGN(marray &Lhs, const marray &Rhs) {                     \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements, typename T = DataT>                 \
  friend typename std::enable_if<Num != 1 && std::is_integral<T>::value,       \
                                 marray &>::type                               \
  operator OPASSIGN(marray &Lhs, const DataT &Rhs) {                           \
    Lhs = Lhs BINOP marray(Rhs);                                               \
    return Lhs;                                                                \
  }

  __SYCL_BINOP(+, +=)
  __SYCL_BINOP(-, -=)
  __SYCL_BINOP(*, *=)
  __SYCL_BINOP(/, /=)

  __SYCL_BINOP_INTEGRAL(%, %=)
  __SYCL_BINOP_INTEGRAL(|, |=)
  __SYCL_BINOP_INTEGRAL(&, &=)
  __SYCL_BINOP_INTEGRAL(^, ^=)
  __SYCL_BINOP_INTEGRAL(>>, >>=)
  __SYCL_BINOP_INTEGRAL(<<, <<=)
#undef __SYCL_BINOP
#undef __SYCL_BINOP_INTEGRAL

#ifdef __SYCL_RELLOGOP
#error "Undefine __SYCL_RELLOGOP macro"
#endif

#ifdef __SYCL_RELLOGOP_INTEGRAL
#error "Undefine __SYCL_RELLOGOP_INTEGRAL macro"
#endif

#define __SYCL_RELLOGOP(RELLOGOP)                                              \
  friend marray<bool, NumElements> operator RELLOGOP(const marray &Lhs,        \
                                                     const marray &Rhs) {      \
    marray<bool, NumElements> Ret;                                             \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] RELLOGOP Rhs[I];                                         \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     (std::is_fundamental<T>::value ||         \
                                      std::is_same<T, half>::value),           \
                                 marray<bool, NumElements>>::type              \
  operator RELLOGOP(const marray &Lhs, const T &Rhs) {                         \
    return Lhs RELLOGOP marray(static_cast<const DataT &>(Rhs));               \
  }

#define __SYCL_RELLOGOP_INTEGRAL(RELLOGOP)                                     \
  template <typename T = DataT>                                                \
  friend typename std::enable_if<std::is_integral<T>::value,                   \
                                 marray<bool, NumElements>>::type              \
  operator RELLOGOP(const marray &Lhs, const marray &Rhs) {                    \
    marray<bool, NumElements> Ret;                                             \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] RELLOGOP Rhs[I];                                         \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if<std::is_convertible<T, DataT>::value &&       \
                                     std::is_integral<T>::value &&             \
                                     std::is_integral<BaseT>::value,           \
                                 marray<bool, NumElements>>::type              \
  operator RELLOGOP(const marray &Lhs, const T &Rhs) {                         \
    return Lhs RELLOGOP marray(static_cast<const DataT &>(Rhs));               \
  }

  __SYCL_RELLOGOP(==)
  __SYCL_RELLOGOP(!=)
  __SYCL_RELLOGOP(>)
  __SYCL_RELLOGOP(<)
  __SYCL_RELLOGOP(>=)
  __SYCL_RELLOGOP(<=)

  __SYCL_RELLOGOP_INTEGRAL(&&)
  __SYCL_RELLOGOP_INTEGRAL(||)

#undef __SYCL_RELLOGOP
#undef __SYCL_RELLOGOP_INTEGRAL

#ifdef __SYCL_UOP
#error "Undefine __SYCL_UOP macro"
#endif

#define __SYCL_UOP(UOP, OPASSIGN)                                              \
  friend marray &operator UOP(marray &Lhs) {                                   \
    Lhs OPASSIGN 1;                                                            \
    return Lhs;                                                                \
  }                                                                            \
  friend marray operator UOP(marray &Lhs, int) {                               \
    marray Ret(Lhs);                                                           \
    Lhs OPASSIGN 1;                                                            \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  // Available only when: dataT != cl_float && dataT != cl_double
  // && dataT != cl_half
  template <typename T = DataT>
  friend typename std::enable_if<std::is_integral<T>::value, marray>::type
  operator~(const marray &Lhs) {
    marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = ~Lhs[I];
    }
    return Ret;
  }

  friend marray<bool, NumElements> operator!(const marray &Lhs) {
    marray<bool, NumElements> Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = !Lhs[I];
    }
    return Ret;
  }

  friend marray operator+(const marray &Lhs) {
    marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = +Lhs[I];
    }
    return Ret;
  }

  friend marray operator-(const marray &Lhs) {
    marray Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = -Lhs[I];
    }
    return Ret;
  }
};

#define __SYCL_MAKE_MARRAY_ALIAS(ALIAS, TYPE, N)                               \
  using ALIAS##N = cl::sycl::marray<TYPE, N>;

#define __SYCL_MAKE_MARRAY_ALIASES_FOR_ARITHMETIC_TYPES(N)                     \
  __SYCL_MAKE_MARRAY_ALIAS(mchar, char, N)                                     \
  __SYCL_MAKE_MARRAY_ALIAS(mshort, short, N)                                   \
  __SYCL_MAKE_MARRAY_ALIAS(mint, int, N)                                       \
  __SYCL_MAKE_MARRAY_ALIAS(mlong, long, N)                                     \
  __SYCL_MAKE_MARRAY_ALIAS(mfloat, float, N)                                   \
  __SYCL_MAKE_MARRAY_ALIAS(mdouble, double, N)                                 \
  __SYCL_MAKE_MARRAY_ALIAS(mhalf, half, N)

#define __SYCL_MAKE_MARRAY_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)            \
  __SYCL_MAKE_MARRAY_ALIAS(mschar, signed char, N)                             \
  __SYCL_MAKE_MARRAY_ALIAS(muchar, unsigned char, N)                           \
  __SYCL_MAKE_MARRAY_ALIAS(mushort, unsigned short, N)                         \
  __SYCL_MAKE_MARRAY_ALIAS(muint, unsigned int, N)                             \
  __SYCL_MAKE_MARRAY_ALIAS(mulong, unsigned long, N)                           \
  __SYCL_MAKE_MARRAY_ALIAS(mlonglong, long long, N)                            \
  __SYCL_MAKE_MARRAY_ALIAS(mulonglong, unsigned long long, N)

#define __SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(N)                        \
  __SYCL_MAKE_MARRAY_ALIASES_FOR_ARITHMETIC_TYPES(N)                           \
  __SYCL_MAKE_MARRAY_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)

__SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(2)
__SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(3)
__SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(4)
__SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(8)
__SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH(16)

#undef __SYCL_MAKE_MARRAY_ALIAS
#undef __SYCL_MAKE_MARRAY_ALIASES_FOR_ARITHMETIC_TYPES
#undef __SYCL_MAKE_MARRAY_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES
#undef __SYCL_MAKE_MARRAY_ALIASES_FOR_MARRAY_LENGTH

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
