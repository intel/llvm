//==----------------- marray.hpp --- Implements marray classes -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>       // for half
#include <sycl/detail/common.hpp> // for ArrayCreator
#include <sycl/half_type.hpp>     // for half

#include <array>       // for array
#include <cstddef>     // for size_t
#include <cstdint>     // for int64_t, int8_t, uint64_t, int16_t
#include <type_traits> // for enable_if_t, remove_const, is_conv...
#include <utility>     // for index_sequence, make_index_sequence

namespace sycl {
inline namespace _V1 {

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

// Trait for checking if an argument type is either convertible to the data
// type or an array of types convertible to the data type.
template <typename DataT, typename T>
struct IsSuitableArgType : std::is_convertible<T, DataT> {};
template <typename DataT, typename T, size_t N>
struct IsSuitableArgType<DataT, marray<T, N>> : std::is_convertible<T, DataT> {
};

// Trait for computing the conjunction of of IsSuitableArgType. The empty type
// list will trivially evaluate to true.
template <typename DataT, typename... ArgTN>
struct AllSuitableArgTypes
    : std::conjunction<IsSuitableArgType<DataT, ArgTN>...> {};

class FlattenMArrayArgHelper {
private:
  // Utility trait for creating an std::array from an marray argument.
  template <typename DataT, typename T, std::size_t... Is>
  static constexpr std::array<DataT, sizeof...(Is)>
  MArrayToArray(const marray<T, sizeof...(Is)> &A, std::index_sequence<Is...>) {
    return {static_cast<DataT>(A.MData[Is])...};
  }

public:
  template <typename DataT, typename T, std::size_t N>
  static constexpr std::array<DataT, N> FlattenMArray(const marray<T, N> &A) {
    return MArrayToArray<DataT>(A, std::make_index_sequence<N>());
  }
  template <typename DataT, typename T>
  static constexpr auto FlattenMArray(const T &A) {
    return std::array<DataT, 1>{static_cast<DataT>(A)};
  }
};

template <typename DataT, typename T> struct FlattenMArrayArg {
  constexpr auto operator()(const T &A) const {
    return FlattenMArrayArgHelper::FlattenMArray<DataT>(A);
  }
};

// Alias for shortening the marray arguments to array converter.
template <typename DataT, typename... ArgTN>
using MArrayArgArrayCreator =
    detail::ArrayCreator<DataT, FlattenMArrayArg, ArgTN...>;

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

  /// FIXME: If the subscript operator is made constexpr these can be removed.
  // Other marray specializations needs to be a friend to access MData.
  template <typename Type_, std::size_t NumElements_> friend class marray;
  // detail::FlattenMArrayArgHelper::MArrayToArray needs to be a friend to
  // access MData.
  friend class detail::FlattenMArrayArgHelper;

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
                detail::AllSuitableArgTypes<DataT, ArgTN...>::value &&
                detail::GetMArrayArgsSize<ArgTN...>::value == NumElements>>
  constexpr marray(const ArgTN &...Args)
      : marray{detail::MArrayArgArrayCreator<DataT, ArgTN...>::Create(Args...),
               std::make_index_sequence<NumElements>()} {}

  constexpr marray(const marray<Type, NumElements> &Rhs) = default;

  constexpr marray(marray<Type, NumElements> &&Rhs) = default;

  // Available only when: NumElements == 1
  template <std::size_t Size = NumElements,
            typename = std::enable_if_t<Size == 1>>
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
  friend typename std::enable_if_t<                                            \
      std::is_convertible_v<DataT, T> &&                                       \
          (std::is_fundamental_v<T> ||                                         \
           std::is_same_v<typename std::remove_const<T>::type, half>),         \
      marray>                                                                  \
  operator BINOP(const marray &Lhs, const T &Rhs) {                            \
    return Lhs BINOP marray(static_cast<DataT>(Rhs));                          \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if_t<                                            \
      std::is_convertible_v<DataT, T> &&                                       \
          (std::is_fundamental_v<T> ||                                         \
           std::is_same_v<typename std::remove_const<T>::type, half>),         \
      marray>                                                                  \
  operator BINOP(const T &Lhs, const marray &Rhs) {                            \
    return marray(static_cast<DataT>(Lhs)) BINOP Rhs;                          \
  }                                                                            \
  friend marray &operator OPASSIGN(marray &Lhs, const marray &Rhs) {           \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements>                                     \
  friend typename std::enable_if_t<Num != 1, marray &> operator OPASSIGN(      \
      marray &Lhs, const DataT &Rhs) {                                         \
    Lhs = Lhs BINOP marray(Rhs);                                               \
    return Lhs;                                                                \
  }

#define __SYCL_BINOP_INTEGRAL(BINOP, OPASSIGN)                                 \
  template <typename T = DataT,                                                \
            typename = std::enable_if_t<std::is_integral_v<T>, marray>>        \
  friend marray operator BINOP(const marray &Lhs, const marray &Rhs) {         \
    marray Ret;                                                                \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if_t<std::is_convertible_v<T, DataT> &&          \
                                       std::is_integral_v<T> &&                \
                                       std::is_integral_v<BaseT>,              \
                                   marray>                                     \
  operator BINOP(const marray &Lhs, const T &Rhs) {                            \
    return Lhs BINOP marray(static_cast<DataT>(Rhs));                          \
  }                                                                            \
  template <typename T, typename BaseT = DataT>                                \
  friend typename std::enable_if_t<std::is_convertible_v<T, DataT> &&          \
                                       std::is_integral_v<T> &&                \
                                       std::is_integral_v<BaseT>,              \
                                   marray>                                     \
  operator BINOP(const T &Lhs, const marray &Rhs) {                            \
    return marray(static_cast<DataT>(Lhs)) BINOP Rhs;                          \
  }                                                                            \
  template <typename T = DataT,                                                \
            typename = std::enable_if_t<std::is_integral_v<T>, marray>>        \
  friend marray &operator OPASSIGN(marray &Lhs, const marray &Rhs) {           \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <std::size_t Num = NumElements, typename T = DataT>                 \
  friend                                                                       \
      typename std::enable_if_t<Num != 1 && std::is_integral_v<T>, marray &>   \
      operator OPASSIGN(marray &Lhs, const DataT &Rhs) {                       \
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
  friend typename std::enable_if_t<std::is_convertible_v<T, DataT> &&          \
                                       (std::is_fundamental_v<T> ||            \
                                        std::is_same_v<T, half>),              \
                                   marray<bool, NumElements>>                  \
  operator RELLOGOP(const marray &Lhs, const T &Rhs) {                         \
    return Lhs RELLOGOP marray(static_cast<const DataT &>(Rhs));               \
  }                                                                            \
  template <typename T>                                                        \
  friend typename std::enable_if_t<std::is_convertible_v<T, DataT> &&          \
                                       (std::is_fundamental_v<T> ||            \
                                        std::is_same_v<T, half>),              \
                                   marray<bool, NumElements>>                  \
  operator RELLOGOP(const T &Lhs, const marray &Rhs) {                         \
    return marray(static_cast<const DataT &>(Lhs)) RELLOGOP Rhs;               \
  }

  __SYCL_RELLOGOP(==)
  __SYCL_RELLOGOP(!=)
  __SYCL_RELLOGOP(>)
  __SYCL_RELLOGOP(<)
  __SYCL_RELLOGOP(>=)
  __SYCL_RELLOGOP(<=)
  __SYCL_RELLOGOP(&&)
  __SYCL_RELLOGOP(||)

#undef __SYCL_RELLOGOP

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
  friend std::enable_if_t<std::is_integral_v<T>, marray>
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

} // namespace _V1
} // namespace sycl
