//==------------ - simd.hpp - DPC++ Explicit SIMD API   --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD vector APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/simd_mask_impl.hpp>
#include <sycl/ext/intel/esimd/detail/simd_obj_impl.hpp>

#include <sycl/ext/intel/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <iostream>
#endif // __SYCL_DEVICE_ONLY__

__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_NS {

/// @addtogroup sycl_esimd_core
/// @{

/// @defgroup sycl_esimd_core_vectors Main vector data types.
/// ESIMD defines the following two main vector data types:
/// - simd. Register-mapped vector of elements.
/// - simd_mask. Register-mapped mask - vector of predicates.

/// @} sycl_esimd_core

/// @addtogroup sycl_esimd_core_vectors
/// @{

/// The main simd vector class.
///
/// A vector of elements, which compiler tries to map to a GPU register.
/// Supports all standard C++ unary and binary operations. See more details
/// in the base class' docs: \c detail::simd_obj_impl.
///
/// @tparam Ty element type. Can be any C++ integer or floating point type or
///    \c sycl::half.
/// @tparam N the number of elements.
template <typename Ty, int N>
class simd : public detail::simd_obj_impl<
                 detail::__raw_t<Ty>, N, simd<Ty, N>,
                 std::enable_if_t<detail::is_valid_simd_elem_type_v<Ty>>> {
  using base_type = detail::simd_obj_impl<detail::__raw_t<Ty>, N, simd<Ty, N>>;

public:
  using base_type::base_type;
  using element_type = Ty;
  using raw_element_type = typename base_type::raw_element_type;
  using raw_vector_type = typename base_type::raw_vector_type;
  static constexpr int length = N;

  /// Implicit conversion constructor from another simd object of the same
  /// length.
  /// Available when \c SimdT is
  /// - instantiation of simd
  /// - has the same number of elements
  /// @tparam SimdT type of the object to convert from
  /// @param RHS object to convert from
  template <typename SimdT,
            class = std::enable_if_t<__ESIMD_DNS::is_simd_type_v<SimdT> &&
                                     (length == SimdT::length)>>
  simd(const SimdT &RHS)
      : base_type(detail::convert_vector<Ty, detail::element_type_t<SimdT>, N>(
            RHS.data())) {
    __esimd_dbg_print(simd(const SimdT &RHS));
  }

  /// Broadcast constructor with conversion. Converts given value to
  /// #element_type and replicates it in all elements.
  /// Available when \c T1 is a valid simd element type.
  /// @tparam T1 broadcast value type
  /// @tparam Val broadcast value
  template <typename T1,
            class = std::enable_if_t<detail::is_valid_simd_elem_type_v<T1>>>
  simd(T1 Val) : base_type(Val) {
    __esimd_dbg_print(simd(T1 Val));
  }

  /// Converts this object to a scalar. Available when
  /// - this object's length is 1
  /// - \c To is a valid simd element type
  /// @tparam To the scalar type
  /// @return this object's single element value converted to the result type.
  template <class To, class T = simd,
            class = sycl::detail::enable_if_t<
                (T::length == 1) && detail::is_valid_simd_elem_type_v<To>>>
  operator To() const {
    __esimd_dbg_print(operator To());
    return detail::convert_scalar<To, element_type>(base_type::data()[0]);
  }

  /// Prefix increment, increments elements of this object.
  /// @return Reference to this object.
  simd &operator++() {
    *this += 1;
    return *this;
  }

  /// Postfix increment.
  /// @return New simd object, whose element values are incremented values of
  /// this object's elements.
  simd operator++(int) {
    simd Ret(base_type::data());
    operator++();
    return Ret;
  }

  /// Prefix decrement, decrements elements of this object.
  /// @return Reference to this object.
  simd &operator--() {
    *this -= 1;
    return *this;
  }

  /// Postfix decrement.
  /// @return New simd object, whose element values are decremented values of
  /// this object's elements.
  simd operator--(int) {
    simd Ret(base_type::data());
    operator--();
    return Ret;
  }

#define __ESIMD_DEF_SIMD_ARITH_UNARY_OP(ARITH_UNARY_OP, ID)                    \
  template <class T1 = Ty> simd operator ARITH_UNARY_OP() const {              \
    static_assert(!std::is_unsigned_v<T1>,                                     \
                  #ARITH_UNARY_OP "doesn't apply to unsigned types");          \
    return simd{detail::vector_unary_op<detail::UnaryOp::ID, T1, N>(           \
        base_type::data())};                                                   \
  }

  /// Unary minus applied to elements of this object.
  __ESIMD_DEF_SIMD_ARITH_UNARY_OP(-, minus)
  /// Unary plus applied to elements of this object.
  __ESIMD_DEF_SIMD_ARITH_UNARY_OP(+, plus)
#undef __ESIMD_DEF_SIMD_ARITH_UNARY_OP
};

/// @} sycl_esimd_core_vectors

/// @addtogroup sycl_esimd_conv
/// @{

/// Covert from a simd object with element type \c From to a simd object with
/// element type \c To.
template <typename To, typename From, int N>
ESIMD_INLINE simd<To, N> convert(const simd<From, N> &val) {
  if constexpr (std::is_same_v<To, From>)
    return val;
  else
    return detail::convert_vector<To, From, N>(val.data());
}
/// @} sycl_esimd_conv

/// @addtogroup sycl_esimd_core_vectors
/// @{

/// Represents a simd mask os size \c N.
/// This is basically an alias of the detail::simd_mask_impl class.
template <int N> using simd_mask = detail::simd_mask_type<N>;

/// @} sycl_esimd_core_vectors

} // namespace __ESIMD_NS
} // __SYCL_INLINE_NAMESPACE(cl)

/// @ingroup sycl_esimd_misc
/// Prints a \c simd object to an output stream.
/// TODO: implemented for host code only.
template <typename Ty, int N>
std::ostream &operator<<(std::ostream &OS, const __ESIMD_NS::simd<Ty, N> &V)
#ifdef __SYCL_DEVICE_ONLY__
    {}
#else
{
  OS << "{";
  for (int I = 0; I < N; I++) {
    OS << V[I];
    if (I < N - 1)
      OS << ",";
  }
  OS << "}";
  return OS;
}
#endif // __SYCL_DEVICE_ONLY__
