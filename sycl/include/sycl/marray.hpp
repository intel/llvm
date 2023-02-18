//==----------------- marray.hpp --- Implements marray classes -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/half_type.hpp>

#include <array>
#include <type_traits>
#include <utility>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

template <typename DataT, std::size_t N> class marray;

namespace detail {

// Helper trait for counting the aggregate number of arguments in a type list,
// expanding marrays.
template <typename... Ts> struct GetMArrayArgsSize;
template <> struct GetMArrayArgsSize<> {
  static constexpr std::size_t value = 0;
};
template <typename T, std::size_t N, typename... Ts>
struct GetMArrayArgsSize<marray<T, N>, Ts...> {
  static constexpr std::size_t value = N + GetMArrayArgsSize<Ts...>::value;
};
template <typename T, typename... Ts> struct GetMArrayArgsSize<T, Ts...> {
  static constexpr std::size_t value = 1 + GetMArrayArgsSize<Ts...>::value;
};

// Helper function for concatenating two std::array.
template <typename T, std::size_t... Is1, std::size_t... Is2>
constexpr std::array<T, sizeof...(Is1) + sizeof...(Is2)>
ConcatArrays(const std::array<T, sizeof...(Is1)> &A1,
             const std::array<T, sizeof...(Is2)> &A2,
             std::index_sequence<Is1...>, std::index_sequence<Is2...>) {
  return {A1[Is1]..., A2[Is2]...};
}
template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> ConcatArrays(const std::array<T, N1> &A1,
                                              const std::array<T, N2> &A2) {
  return ConcatArrays(A1, A2, std::make_index_sequence<N1>(),
                      std::make_index_sequence<N2>());
}

// Utility trait for creating an std::array from an marray.
template <typename DataT, typename T, std::size_t... Is>
constexpr std::array<T, sizeof...(Is)>
MArrayToArray(const marray<T, sizeof...(Is)> &A, std::index_sequence<Is...>) {
  return {static_cast<DataT>(A.MData[Is])...};
}
template <typename DataT, typename T, std::size_t N>
constexpr std::array<T, N> MArrayToArray(const marray<T, N> &A) {
  return MArrayToArray<DataT>(A, std::make_index_sequence<N>());
}

// Utility for creating an std::array from a arguments of either types
// convertible to DataT or marrays of a type convertible to DataT.
template <typename DataT, typename... ArgTN> struct ArrayCreator;
template <typename DataT, typename ArgT, typename... ArgTN>
struct ArrayCreator<DataT, ArgT, ArgTN...> {
  static constexpr std::array<DataT, GetMArrayArgsSize<ArgT, ArgTN...>::value>
  Create(const ArgT &Arg, const ArgTN &...Args) {
    return ConcatArrays(std::array<DataT, 1>{static_cast<DataT>(Arg)},
                        ArrayCreator<DataT, ArgTN...>::Create(Args...));
  }
};
template <typename DataT, typename T, std::size_t N, typename... ArgTN>
struct ArrayCreator<DataT, marray<T, N>, ArgTN...> {
  static constexpr std::array<DataT,
                              GetMArrayArgsSize<marray<T, N>, ArgTN...>::value>
  Create(const marray<T, N> &Arg, const ArgTN &...Args) {
    return ConcatArrays(MArrayToArray<DataT>(Arg),
                        ArrayCreator<DataT, ArgTN...>::Create(Args...));
  }
};
template <typename DataT> struct ArrayCreator<DataT> {
  static constexpr std::array<DataT, 0> Create() {
    return std::array<DataT, 0>{};
  }
};
} // namespace detail

/// Provides a cross-platform math array class template that works on
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

  // Trait for checking if an argument type is either convertible to the data
  // type or an array of types convertible to the data type.
  template <typename T>
  struct IsSuitableArgType : std::is_convertible<T, DataT> {};
  template <typename T, size_t N>
  struct IsSuitableArgType<marray<T, N>> : std::is_convertible<T, DataT> {};

  // Trait for computing the conjunction of of IsSuitableArgType. The empty type
  // list will trivially evaluate to true.
  template <typename... ArgTN>
  struct AllSuitableArgTypes : std::conjunction<IsSuitableArgType<ArgTN>...> {};

  // FIXME: MArrayToArray needs to be a friend to access MData. If the subscript
  //        operator is made constexpr this can be removed.
  template <typename, typename T, std::size_t... Is>
  friend constexpr std::array<T, sizeof...(Is)>
  detail::MArrayToArray(const marray<T, sizeof...(Is)> &,
                        std::index_sequence<Is...>);

  constexpr void initialize_data(const Type &Arg) {
    for (size_t i = 0; i < NumElements; ++i) {
      MData[i] = Arg;
    }
  }

  template <size_t... Is>
  constexpr marray(const std::array<DataT, NumElements> &Arr,
                   std::index_sequence<Is...>)
      : MData{Arr[Is]...} {}

public:
  constexpr marray() : MData{} {}

  explicit constexpr marray(const Type &Arg) : MData{Arg} {
    initialize_data(Arg);
  }

  template <typename... ArgTN,
            typename = std::enable_if_t<
                AllSuitableArgTypes<ArgTN...>::value &&
                detail::GetMArrayArgsSize<ArgTN...>::value == NumElements>>
  constexpr marray(const ArgTN &...Args)
      : marray{detail::ArrayCreator<DataT, ArgTN...>::Create(Args...),
               std::make_index_sequence<NumElements>()} {}

  constexpr marray(const marray<Type, NumElements> &Rhs) = default;

  constexpr marray(marray<Type, NumElements> &&Rhs) = default;

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

  marray &operator=(const marray<Type, NumElements> &Rhs) = default;

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
  template <typename T = DataT,                                                \
            typename = std::enable_if<std::is_integral<T>::value, marray>>     \
  friend marray operator BINOP(const marray &Lhs, const marray &Rhs) {         \
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
  template <typename T = DataT,                                                \
            typename = std::enable_if<std::is_integral<T>::value, marray>>     \
  friend marray &operator OPASSIGN(marray &Lhs, const marray &Rhs) {           \
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
  using ALIAS##N = sycl::marray<TYPE, N>;

#define __SYCL_MAKE_MARRAY_ALIASES_FOR_ARITHMETIC_TYPES(N)                     \
  __SYCL_MAKE_MARRAY_ALIAS(mbool, bool, N)                                     \
  __SYCL_MAKE_MARRAY_ALIAS(mchar, std::int8_t, N)                              \
  __SYCL_MAKE_MARRAY_ALIAS(mshort, std::int16_t, N)                            \
  __SYCL_MAKE_MARRAY_ALIAS(mint, std::int32_t, N)                              \
  __SYCL_MAKE_MARRAY_ALIAS(mlong, std::int64_t, N)                             \
  __SYCL_MAKE_MARRAY_ALIAS(mlonglong, std::int64_t, N)                         \
  __SYCL_MAKE_MARRAY_ALIAS(mfloat, float, N)                                   \
  __SYCL_MAKE_MARRAY_ALIAS(mdouble, double, N)                                 \
  __SYCL_MAKE_MARRAY_ALIAS(mhalf, half, N)

// FIXME: schar, longlong and ulonglong aliases are not defined by SYCL 2020
//        spec, but they are preserved in SYCL 2020 mode, because SYCL-CTS is
//        still using them.
//        See KhronosGroup/SYCL-CTS#446 and KhronosGroup/SYCL-Docs#335
#define __SYCL_MAKE_MARRAY_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)            \
  __SYCL_MAKE_MARRAY_ALIAS(mschar, std::int8_t, N)                             \
  __SYCL_MAKE_MARRAY_ALIAS(muchar, std::uint8_t, N)                            \
  __SYCL_MAKE_MARRAY_ALIAS(mushort, std::uint16_t, N)                          \
  __SYCL_MAKE_MARRAY_ALIAS(muint, std::uint32_t, N)                            \
  __SYCL_MAKE_MARRAY_ALIAS(mulong, std::uint64_t, N)                           \
  __SYCL_MAKE_MARRAY_ALIAS(mulonglong, std::uint64_t, N)

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

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
