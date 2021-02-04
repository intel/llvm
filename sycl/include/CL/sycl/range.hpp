//==----------- range.hpp --- SYCL iteration range -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/array.hpp>
#include <CL/sycl/detail/helpers.hpp>

#include <stdexcept>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class id;

/// Defines the iteration domain of either a single work-group in a parallel
/// dispatch, or the overall dimensions of the dispatch.
///
/// \ingroup sycl_api
template <int dimensions = 1> class range : public detail::array<dimensions> {
  static_assert(dimensions >= 1 && dimensions <= 3,
                "range can only be 1, 2, or 3 dimensional.");
  using base = detail::array<dimensions>;
  template <typename N, typename T>
  using IntegralType = detail::enable_if_t<std::is_integral<N>::value, T>;

public:
  /* The following constructor is only available in the range class
  specialization where: dimensions==1 */
  template <int N = dimensions>
  range(typename detail::enable_if_t<(N == 1), size_t> dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: dimensions==2 */
  template <int N = dimensions>
  range(typename detail::enable_if_t<(N == 2), size_t> dim0, size_t dim1)
      : base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: dimensions==3 */
  template <int N = dimensions>
  range(typename detail::enable_if_t<(N == 3), size_t> dim0, size_t dim1,
        size_t dim2)
      : base(dim0, dim1, dim2) {}

  explicit operator id<dimensions>() const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < dimensions; ++i) {
      size *= this->get(i);
    }
    return size;
  }

  range(const range<dimensions> &rhs) = default;
  range(range<dimensions> &&rhs) = default;
  range<dimensions> &operator=(const range<dimensions> &rhs) = default;
  range<dimensions> &operator=(range<dimensions> &&rhs) = default;
  range() = delete;

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT(op)                                                     \
  range<dimensions> operator op(const range<dimensions> &rhs) const {          \
    range<dimensions> result(*this);                                           \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs.common_array[i];   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  IntegralType<T, range<dimensions>> operator op(const T &rhs) const {         \
    range<dimensions> result(*this);                                           \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs;                   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend IntegralType<T, range<dimensions>> operator op(                       \
      const T &lhs, const range<dimensions> &rhs) {                            \
    range<dimensions> result(rhs);                                             \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }

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

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  range<dimensions> &operator op(const range<dimensions> &rhs) {               \
    for (int i = 0; i < dimensions; ++i) {                                     \
      this->common_array[i] op rhs[i];                                         \
    }                                                                          \
    return *this;                                                              \
  }                                                                            \
  range<dimensions> &operator op(const size_t &rhs) {                          \
    for (int i = 0; i < dimensions; ++i) {                                     \
      this->common_array[i] op rhs;                                            \
    }                                                                          \
    return *this;                                                              \
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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
