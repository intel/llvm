//==----------- range.hpp --- SYCL iteration range -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/array.hpp>   // for array
#include <sycl/detail/helpers.hpp> // for Builder

#include <array>       // for array
#include <stddef.h>    // for size_t
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
template <int Dimensions> class id;

/// Defines the iteration domain of either a single work-group in a parallel
/// dispatch, or the overall Dimensions of the dispatch.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class range : public detail::array<Dimensions> {
public:
  static constexpr int dimensions = Dimensions;

private:
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "range can only be 1, 2, or 3 Dimensional.");
  using base = detail::array<Dimensions>;
  template <typename N, typename T>
  using IntegralType = std::enable_if_t<std::is_integral_v<N>, T>;

public:
  /* The following constructor is only available in the range class
  specialization where: Dimensions==1 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 1), size_t> dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==2 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 2), size_t> dim0, size_t dim1)
      : base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==3 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 3), size_t> dim0, size_t dim1,
        size_t dim2)
      : base(dim0, dim1, dim2) {}

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < Dimensions; ++i) {
      size *= this->common_array[i];
    }
    return size;
  }

  range(const range<Dimensions> &rhs) = default;
  range(range<Dimensions> &&rhs) = default;
  range<Dimensions> &operator=(const range<Dimensions> &rhs) = default;
  range<Dimensions> &operator=(range<Dimensions> &&rhs) = default;
  range() = default;

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs,           \
                                       const range<Dimensions> &rhs) {         \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs.common_array[i];     \
    }                                                                          \
    return result;                                                             \
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  // Enable operators with integral types only
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <typename T>                                                        \
  friend IntegralType<T, range<Dimensions>> operator op(                       \
      const range<Dimensions> &lhs, const T &rhs) {                            \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend IntegralType<T, range<Dimensions>> operator op(                       \
      const T &lhs, const range<Dimensions> &rhs) {                            \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs,           \
                                       const size_t &rhs) {                    \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  friend range<Dimensions> operator op(const size_t &lhs,                      \
                                       const range<Dimensions> &rhs) {         \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)
  __SYCL_GEN_OPT(*)
  __SYCL_GEN_OPT(/)
  __SYCL_GEN_OPT(%)
  __SYCL_GEN_OPT(<<)
  __SYCL_GEN_OPT(>>)
  __SYCL_GEN_OPT(&)
  __SYCL_GEN_OPT(|)
  __SYCL_GEN_OPT(^)
  __SYCL_GEN_OPT(&&)
  __SYCL_GEN_OPT(||)
  __SYCL_GEN_OPT(<)
  __SYCL_GEN_OPT(>)
  __SYCL_GEN_OPT(<=)
  __SYCL_GEN_OPT(>=)

#undef __SYCL_GEN_OPT
#undef __SYCL_GEN_OPT_BASE

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> &operator op(range<Dimensions> &lhs,                \
                                        const range<Dimensions> &rhs) {        \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs[i];                                           \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  friend range<Dimensions> &operator op(range<Dimensions> &lhs,                \
                                        const size_t &rhs) {                   \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs;                                              \
    }                                                                          \
    return lhs;                                                                \
  }

  __SYCL_GEN_OPT(+=)
  __SYCL_GEN_OPT(-=)
  __SYCL_GEN_OPT(*=)
  __SYCL_GEN_OPT(/=)
  __SYCL_GEN_OPT(%=)
  __SYCL_GEN_OPT(<<=)
  __SYCL_GEN_OPT(>>=)
  __SYCL_GEN_OPT(&=)
  __SYCL_GEN_OPT(|=)
  __SYCL_GEN_OPT(^=)

#undef __SYCL_GEN_OPT

// OP is unary +, -
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> operator op(const range<Dimensions> &rhs) {         \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = (op rhs.common_array[i]);                       \
    }                                                                          \
    return result;                                                             \
  }

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)

#undef __SYCL_GEN_OPT

// OP is prefix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> &operator op(range<Dimensions> &rhs) {              \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op rhs.common_array[i];                                                  \
    }                                                                          \
    return rhs;                                                                \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

// OP is postfix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> operator op(range<Dimensions> &lhs, int) {          \
    range<Dimensions> old_lhs(lhs);                                            \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op lhs.common_array[i];                                                  \
    }                                                                          \
    return old_lhs;                                                            \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

private:
  friend class handler;
  friend class detail::Builder;

  // Adjust the first dim of the range
  void set_range_dim0(const size_t dim0) { this->common_array[0] = dim0; }
};

#ifdef __cpp_deduction_guides
range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;
#endif

namespace detail {
// XPTI helpers for creating array from a range.
inline std::array<size_t, 3> rangeToArray(const range<3> &r) {
  return {r[0], r[1], r[2]};
}
inline std::array<size_t, 3> rangeToArray(const range<2> &r) {
  return {r[0], r[1], 0};
}
inline std::array<size_t, 3> rangeToArray(const range<1> &r) {
  return {r[0], 0, 0};
}
} // namespace detail

} // namespace _V1
} // namespace sycl
