//==----------- id.hpp --- SYCL iteration id -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/array.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/range.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class range;
template <int dimensions, bool with_offset> class item;

/// A unique identifier of an item in an index space.
///
/// \ingroup sycl_api
template <int dimensions = 1> class id : public detail::array<dimensions> {
private:
  using base = detail::array<dimensions>;
  static_assert(dimensions >= 1 && dimensions <= 3,
                "id can only be 1, 2, or 3 dimensional.");
  template <int N, int val, typename T>
  using ParamTy = detail::enable_if_t<(N == val), T>;

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator __private_class(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class __private_class;

  template <typename N, typename T>
  using EnableIfIntegral = detail::enable_if_t<std::is_integral<N>::value, T>;
  template <bool B, typename T>
  using EnableIfT = detail::conditional_t<B, T, __private_class>;
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

public:
  id() = default;

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==1 */
  template <int N = dimensions> id(ParamTy<N, 1, size_t> dim0) : base(dim0) {}

  template <int N = dimensions>
  id(ParamTy<N, 1, const range<dimensions>> &range_size)
      : base(range_size.get(0)) {}

  template <int N = dimensions, bool with_offset = true>
  id(ParamTy<N, 1, const item<dimensions, with_offset>> &item)
      : base(item.get_id(0)) {}

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==2 */
  template <int N = dimensions>
  id(ParamTy<N, 2, size_t> dim0, size_t dim1) : base(dim0, dim1) {}

  template <int N = dimensions>
  id(ParamTy<N, 2, const range<dimensions>> &range_size)
      : base(range_size.get(0), range_size.get(1)) {}

  template <int N = dimensions, bool with_offset = true>
  id(ParamTy<N, 2, const item<dimensions, with_offset>> &item)
      : base(item.get_id(0), item.get_id(1)) {}

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==3 */
  template <int N = dimensions>
  id(ParamTy<N, 3, size_t> dim0, size_t dim1, size_t dim2)
      : base(dim0, dim1, dim2) {}

  template <int N = dimensions>
  id(ParamTy<N, 3, const range<dimensions>> &range_size)
      : base(range_size.get(0), range_size.get(1), range_size.get(2)) {}

  template <int N = dimensions, bool with_offset = true>
  id(ParamTy<N, 3, const item<dimensions, with_offset>> &item)
      : base(item.get_id(0), item.get_id(1), item.get_id(2)) {}

  explicit operator range<dimensions>() const {
    range<dimensions> result(
        detail::InitializedVal<dimensions, range>::template get<0>());
    for (int i = 0; i < dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  /* Template operator is not allowed because it disables further type
   * conversion. For example, the next code will not work in case of template
   * conversion:
   * int a = id<1>(value); */

  __SYCL_ALWAYS_INLINE operator EnableIfT<(dimensions == 1), size_t>() const {
    size_t Result = this->common_array[0];
    __SYCL_ASSUME_INT(Result);
    return Result;
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

// OP is: ==, !=
#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  using detail::array<dimensions>::operator==;
#if __cpp_impl_three_way_comparison < 201907
  using detail::array<dimensions>::operator!=;
#endif

  /* Enable operators with integral types.
   * Template operators take precedence than type conversion. In the case of
   * non-template operators, ambiguity appears: "id op size_t" may refer
   * "size_t op size_t" and "id op size_t". In case of template operators it
   * will be "id op size_t"*/
#define __SYCL_GEN_OPT(op)                                                     \
  template <typename T>                                                        \
  EnableIfIntegral<T, bool> operator op(const T &rhs) const {                  \
    if (this->common_array[0] != rhs)                                          \
      return false op true;                                                    \
    return true op true;                                                       \
  }                                                                            \
  template <typename T>                                                        \
  friend EnableIfIntegral<T, bool> operator op(const T &lhs,                   \
                                               const id<dimensions> &rhs) {    \
    if (lhs != rhs.common_array[0])                                            \
      return false op true;                                                    \
    return true op true;                                                       \
  }

  __SYCL_GEN_OPT(==)
  __SYCL_GEN_OPT(!=)

#undef __SYCL_GEN_OPT

#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  id<dimensions> operator op(const id<dimensions> &rhs) const {                \
    id<dimensions> result;                                                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs.common_array[i];   \
    }                                                                          \
    return result;                                                             \
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
// Enable operators with integral types only
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <typename T>                                                        \
  EnableIfIntegral<T, id<dimensions>> operator op(const T &rhs) const {        \
    id<dimensions> result;                                                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs;                   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend EnableIfIntegral<T, id<dimensions>> operator op(                      \
      const T &lhs, const id<dimensions> &rhs) {                               \
    id<dimensions> result;                                                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  id<dimensions> operator op(const size_t &rhs) const {                        \
    id<dimensions> result;                                                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
      result.common_array[i] = this->common_array[i] op rhs;                   \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  friend id<dimensions> operator op(const size_t &lhs,                         \
                                    const id<dimensions> &rhs) {               \
    id<dimensions> result;                                                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
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
  id<dimensions> &operator op(const id<dimensions> &rhs) {                     \
    for (int i = 0; i < dimensions; ++i) {                                     \
      this->common_array[i] op rhs.common_array[i];                            \
    }                                                                          \
    return *this;                                                              \
  }                                                                            \
  id<dimensions> &operator op(const size_t &rhs) {                             \
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
  void set_allowed_range(range<dimensions> rnwi) { (void)rnwi[0]; }
};

namespace detail {
template <int dimensions>
size_t getOffsetForId(range<dimensions> Range, id<dimensions> Id,
                      id<dimensions> Offset) {
  size_t offset = 0;
  for (int i = 0; i < dimensions; ++i)
    offset = offset * Range[i] + Offset[i] + Id[i];
  return offset;
}

inline id<1> getDelinearizedId(const range<1> &, size_t Index) {
  return {Index};
}

inline id<2> getDelinearizedId(const range<2> &Range, size_t Index) {
  size_t X = Index % Range[1];
  size_t Y = Index / Range[1];
  return {Y, X};
}

inline id<3> getDelinearizedId(const range<3> &Range, size_t Index) {
  size_t D1D2 = Range[1] * Range[2];
  size_t Z = Index / D1D2;
  size_t ZRest = Index % D1D2;
  size_t Y = ZRest / Range[2];
  size_t X = ZRest % Range[2];
  return {Z, Y, X};
}
} // namespace detail

// C++ feature test macros are supported by all supported compilers
// with the exception of MSVC 1914. It doesn't support deduction guides.
#ifdef __cpp_deduction_guides
id(size_t)->id<1>;
id(size_t, size_t)->id<2>;
id(size_t, size_t, size_t)->id<3>;
#endif

namespace detail {
template <int Dims> id<Dims> store_id(const id<Dims> *i) {
  return get_or_store(i);
}
} // namespace detail

template <int Dims>
__SYCL_DEPRECATED("use sycl::ext::oneapi::experimental::this_id() instead")
id<Dims> this_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::Builder::getElement(detail::declptr<id<Dims>>());
#else
  return detail::store_id<Dims>(nullptr);
#endif
}

namespace ext {
namespace oneapi {
namespace experimental {
template <int Dims> id<Dims> this_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(detail::declptr<id<Dims>>());
#else
  return sycl::detail::store_id<Dims>(nullptr);
#endif
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
