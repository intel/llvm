//===- marray.hpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <sycl/marray.hpp>
#include <sycl/types.hpp>

namespace sycl {
inline namespace _V1 {

template <typename T> using marray_data = sycl::detail::vec_helper<T>;

template <typename T>
using marray_data_t = typename detail::vec_helper<T>::RetType;

template <typename T, std::size_t NumElements>
class marray<sycl::ext::oneapi::experimental::complex<T>, NumElements> {
private:
  using ComplexDataT = sycl::ext::oneapi::experimental::complex<T>;

public:
  using value_type = ComplexDataT;
  using reference = ComplexDataT &;
  using const_reference = const ComplexDataT &;
  using iterator = ComplexDataT *;
  using const_iterator = const ComplexDataT *;

private:
  value_type MData[NumElements];

  template <size_t... Is>
  constexpr marray(const std::array<value_type, NumElements> &Arr,
                   std::index_sequence<Is...>)
      : MData{Arr[Is]...} {}

  // detail::FlattenMArrayArgHelper::MArrayToArray needs to have access to
  // MData.
  // FIXME: If the subscript operator is made constexpr this can be removed.
  friend class detail::FlattenMArrayArgHelper;

public:
  constexpr marray() : MData{} {};

  explicit constexpr marray(const value_type &arg)
      : marray{sycl::detail::RepeatValue<NumElements>(
                   static_cast<marray_data_t<value_type>>(arg)),
               std::make_index_sequence<NumElements>()} {}

  template <
      typename... ArgTN,
      typename = std::enable_if_t<
          sycl::detail::AllSuitableArgTypes<value_type, ArgTN...>::value &&
          sycl::detail::GetMArrayArgsSize<ArgTN...>::value == NumElements>>
  constexpr marray(const ArgTN &...Args)
      : marray{
            sycl::detail::MArrayArgArrayCreator<value_type, ArgTN...>::Create(
                Args...),
            std::make_index_sequence<NumElements>()} {}

  constexpr marray(const marray<value_type, NumElements> &rhs) = default;
  constexpr marray(marray<value_type, NumElements> &&rhs) = default;

  // Available only when: NumElements == 1
  template <std::size_t N = NumElements,
            typename = std::enable_if_t<N == 1, value_type>>
  operator value_type() const {
    return MData[0];
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  // subscript operator
  reference operator[](std::size_t i) { return MData[i]; }
  const_reference operator[](std::size_t i) const { return MData[i]; }

  marray &operator=(const marray<value_type, NumElements> &rhs) = default;
  marray &operator=(const value_type &rhs) {
    for (std::size_t i = 0; i < NumElements; ++i) {
      MData[i] = rhs;
    }
    return *this;
  }

  // iterator functions
  iterator begin() { return MData; }
  const_iterator begin() const { return MData; }

  iterator end() { return MData + NumElements; }
  const_iterator end() const { return MData + NumElements; }

#ifdef MARRAY_CPLX_OP
#error "Multiple definition of MARRAY_CPLX_OP"
#endif

  // MARRAY_CPLX_OP is: +, -, *, /
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) {            \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs[i];                                               \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const marray &lhs, const value_type &rhs) {        \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs;                                                  \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const value_type &lhs, const marray &rhs) {        \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs op rhs[i];                                                  \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(+)
  MARRAY_CPLX_OP(-)
  MARRAY_CPLX_OP(*)
  MARRAY_CPLX_OP(/)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: %
  friend marray operator%(const marray &lhs, const marray &rhs) = delete;
  friend marray operator%(const marray &lhs, const value_type &rhs) = delete;
  friend marray operator%(const value_type &lhs, const marray &rhs) = delete;

  // MARRAY_CPLX_OP is: +=, -=, *=, /=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray & lhs, const marray & rhs) {               \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs[i];                                                        \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
                                                                               \
  friend marray &operator op(marray & lhs, const value_type & rhs) {           \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  friend marray &operator op(value_type & lhs, const marray & rhs) {           \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }

  MARRAY_CPLX_OP(+=)
  MARRAY_CPLX_OP(-=)
  MARRAY_CPLX_OP(*=)
  MARRAY_CPLX_OP(/=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: %=
  friend marray &operator%=(marray &lhs, const marray &rhs) = delete;
  friend marray &operator%=(marray &lhs, const value_type &rhs) = delete;
  friend marray &operator%=(value_type &lhs, const marray &rhs) = delete;

// MARRAY_CPLX_OP is: ++, --
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(marray &lhs, int) = delete;                        \
  friend marray &operator op(marray & rhs) = delete;

  MARRAY_CPLX_OP(++)
  MARRAY_CPLX_OP(--)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: unary +, unary -
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<value_type, NumElements> operator op(                          \
      const marray<value_type, NumElements> &rhs) {                            \
    marray<value_type, NumElements> rtn;                                       \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = op rhs[i];                                                      \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(+)
  MARRAY_CPLX_OP(-)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &, |, ^
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const value_type &rhs) = delete;

  MARRAY_CPLX_OP(&)
  MARRAY_CPLX_OP(|)
  MARRAY_CPLX_OP(^)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &=, |=, ^=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray & lhs, const marray & rhs) = delete;       \
  friend marray &operator op(marray & lhs, const value_type & rhs) = delete;   \
  friend marray &operator op(value_type & lhs, const marray & rhs) = delete;

  MARRAY_CPLX_OP(&=)
  MARRAY_CPLX_OP(|=)
  MARRAY_CPLX_OP(^=)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &&, ||
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray & lhs,             \
                                               const marray & rhs) = delete;   \
  friend marray<bool, NumElements> operator op(                                \
      const marray & lhs, const value_type & rhs) = delete;                    \
  friend marray<bool, NumElements> operator op(const value_type & lhs,         \
                                               const marray & rhs) = delete;

  MARRAY_CPLX_OP(&&)
  MARRAY_CPLX_OP(||)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: <<, >>
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const value_type &rhs) =        \
      delete;                                                                  \
  friend marray operator op(const value_type &lhs, const marray &rhs) = delete;

  MARRAY_CPLX_OP(<<)
  MARRAY_CPLX_OP(>>)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: <<=, >>=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray & lhs, const marray & rhs) = delete;       \
  friend marray &operator op(marray & lhs, const value_type & rhs) = delete;

  MARRAY_CPLX_OP(<<=)
  MARRAY_CPLX_OP(>>=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: ==, !=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray & lhs,             \
                                               const marray & rhs) {           \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs[i];                                               \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const marray & lhs,             \
                                               const value_type & rhs) {       \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs;                                                  \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const value_type & lhs,         \
                                               const marray & rhs) {           \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs op rhs[i];                                                  \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(==)
  MARRAY_CPLX_OP(!=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: <, >, <=, >=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray & lhs,             \
                                               const marray & rhs) = delete;   \
  friend marray<bool, NumElements> operator op(                                \
      const marray & lhs, const value_type & rhs) = delete;                    \
  friend marray<bool, NumElements> operator op(const value_type & lhs,         \
                                               const marray & rhs) = delete;

  MARRAY_CPLX_OP(<);
  MARRAY_CPLX_OP(>);
  MARRAY_CPLX_OP(<=);
  MARRAY_CPLX_OP(>=);

#undef MARRAY_CPLX_OP

  friend marray operator~(const marray &v) = delete;

  friend marray<bool, NumElements> operator!(const marray &v) = delete;
};

} // namespace _V1
} // namespace sycl
