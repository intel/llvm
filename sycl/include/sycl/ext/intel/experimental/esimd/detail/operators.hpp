//==-------------- operators.hpp - DPC++ Explicit SIMD API -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Binary operator definitions for ESIMD types.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/simd_obj_impl.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/simd_view_impl.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

#include <sycl/ext/intel/experimental/esimd/simd.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

// Put operators into the ESIMD detail namespace to make argument-dependent
// lookup find these operators instead of those defined in e.g. sycl namespace
// (which would stop further lookup, leaving just non-viable sycl::operator <
// etc. on the table).

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {
// clang-format off
/// @addtogroup sycl_esimd_core
/// @{

/// @defgroup sycl_esimd_core_binops C++ binary operators overloads for ESIMD.
/// Standard C++ binary operators overloads applicable to \c simd_obj_impl
/// derivatives - \c simd , \c simd_mask , \c simd_view and their combinations.
/// The following overloads are defined:
///
/// - \c simd_obj_impl global operators:
///     + bitwise logic and arithmetic operators
///          * \c simd_obj_impl BINOP \c simd_obj_impl
///          * \c simd_obj_impl BINOP SCALAR
///          * SCALAR BINOP \c simd_obj_impl
///     + comparison operators
///          * \c simd_obj_impl CMPOP \c simd_obj_impl
///          * \c simd_obj_impl CMPOP SCALAR
///          * SCALAR CMPOP \c simd_obj_impl
/// - \c simd_view global operators
///     + bitwise logic and arithmetic operators
///          * \c simd_view BINOP \c simd_view
///          * \c simd* BINOP \c simd_view<simd*...>
///          * \c simd_view<simd*...> BINOP \c simd*
///          * SCALAR BINOP \c simd_view
///          * \c simd_view BINOP SCALAR
/// - comparison operators
///     * \c simd_view CMPOP \c simd_view
///     * \c simd_view CMPOP \c simd_obj_impl
///     * \c simd_obj_impl CMPOP \c simd_view
///     * \c simd_view CMPOP SCALAR
///     * SCALAR CMPOP \c simd_view
///
/// Some operations are enabled only for particular element type and/or simd
/// object type (simd or simd_mask):
/// - bitwise logic operations - for integral element types (both simd and
///   simd_mask)
/// - bit shift operations and and '%' - for the simd type (not for simd_mask)
///   with integral element types
/// - arithmetic binary operations - for the simd type (not for simd_mask)
/// In all cases, when an operation has a simd_view and a simd_obj_impl's
/// subclass objects as operands, it is enabled only when:
/// - simd_view's base type matches the simd object operand. I.e. only
///   { simd_view<simd, ...>, simd } and { simd_view<simd_mask,...>, simd_mask }
///   pairs are enabled (with any order of operand types).
/// - simd_view's value length matches the length of the simd object operand
///
/// The tables below provides more details about supported overloads.
///
/// Binary operators:
/// |              |simd/simd_view (integer)|simd/simd_view (floating point)|simd_mask|
/// |--------------|:----------------------:|:-----------------------------:|:-------:|
/// | <tt>+   </tt>|      +                 |          +                    |         |
/// | <tt>-   </tt>|      +                 |          +                    |         |
/// | <tt>*   </tt>|      +                 |          +                    |         |
/// | <tt>/   </tt>|      +                 |          +                    |         |
/// | <tt>%   </tt>|      +                 |                               |         |
/// | <tt>\<\<</tt>|      +                 |                               |         |
/// | <tt>\>\></tt>|      +                 |                               |         |
/// | <tt>^   </tt>|      +                 |                               |    +    |
/// | <tt>\|  </tt>|      +                 |                               |    +    |
/// | <tt>\&  </tt>|      +                 |                               |    +    |
/// | <tt>\|\|</tt>|                        |                               |    +    |
/// | <tt>\&\&</tt>|                        |                               |    +    |
///
/// Comparison operators
/// |              |simd/simd_view (integer)|simd/simd_view (floating point)|simd_mask|
/// |--------------|:----------------------:|:-----------------------------:|:-------:|
/// | <tt>== </tt> |      +                 |         +                     |    +    |
/// | <tt>!= </tt> |      +                 |         +                     |    +    |
/// | <tt>\< </tt> |      +                 |         +                     |         |
/// | <tt>\> </tt> |      +                 |         +                     |         |
/// | <tt>\<=</tt> |      +                 |         +                     |         |
/// | <tt>\>=</tt> |      +                 |         +                     |         |
/// @}
// clang-format on

////////////////////////////////////////////////////////////////////////////////
// simd_obj_impl global operators
////////////////////////////////////////////////////////////////////////////////

// ========= simd_obj_impl bitwise logic and arithmetic operators

#define __ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(BINOP, BINOP_ID, COND)                \
                                                                               \
  /* simd_obj_impl BINOP simd_obj_impl */                                      \
  template <class T1, class T2, int N, template <class, int> class SimdT,      \
            class SimdTx = SimdT<T1, N>, class = std::enable_if_t<COND>>       \
  inline auto operator BINOP(                                                  \
      const __SEIEED::simd_obj_impl<__raw_t<T1>, N, SimdT<T1, N>> &LHS,        \
      const __SEIEED::simd_obj_impl<__raw_t<T2>, N, SimdT<T2, N>> &RHS) {      \
    if constexpr (__SEIEED::is_simd_type_v<SimdT<T1, N>>) {                    \
      using PromotedT = __SEIEED::computation_type_t<T1, T2>;                  \
      /* vector_binary_op returns SimdT<PromotedT, N>::raw_vector_type */      \
      SimdT<PromotedT, N> Res = vector_binary_op<BINOP_ID, PromotedT, N>(      \
          __SEIEED::convert_vector<PromotedT, T1, N>(LHS.data()),              \
          __SEIEED::convert_vector<PromotedT, T2, N>(RHS.data()));             \
      return Res;                                                              \
    } else {                                                                   \
      /* for SimdT=simd_mask_impl T1 and T2 are both equal to                  \
       * simd_mask_elem_type */                                                \
      return SimdT<T1, N>(LHS.data() BINOP RHS.data());                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  /* simd_obj_impl BINOP SCALAR */                                             \
  template <class T1, int N1, template <class, int> class SimdT1, class T2,    \
            class SimdTx = SimdT1<T1, N1>, class = std::enable_if_t<COND>>     \
  inline auto operator BINOP(                                                  \
      const __SEIEED::simd_obj_impl<__raw_t<T1>, N1, SimdT1<T1, N1>> &LHS,     \
      T2 RHS) {                                                                \
    if constexpr (__SEIEED::is_simd_type_v<SimdT1<T1, N1>>) {                  \
      /* convert the SCALAR to vector type and reuse the basic operation over  \
       * simd objects */                                                       \
      return LHS BINOP SimdT1<T2, N1>(RHS);                                    \
    } else {                                                                   \
      /* SimdT1 is a mask, T1 is mask element type - convert RHS implicitly to \
       * T1 */                                                                 \
      return LHS BINOP SimdT1<T1, N1>(RHS);                                    \
    }                                                                          \
  }                                                                            \
                                                                               \
  /* SCALAR BINOP simd_obj_impl */                                             \
  template <class T1, class T2, int N2, template <class, int> class SimdT2,    \
            class SimdTx = SimdT2<T2, N2>, class = std::enable_if_t<COND>>     \
  inline auto operator BINOP(                                                  \
      T1 LHS,                                                                  \
      const __SEIEED::simd_obj_impl<__raw_t<T2>, N2, SimdT2<T2, N2>> &RHS) {   \
    if constexpr (__SEIEED::is_simd_type_v<SimdT2<T2, N2>>) {                  \
      /* convert the SCALAR to vector type and reuse the basic operation over  \
       * simd objects */                                                       \
      return SimdT2<T1, N2>(LHS) BINOP RHS;                                    \
    } else {                                                                   \
      /* simd_mask_case */                                                     \
      return SimdT2<T2, N2>(LHS) BINOP RHS;                                    \
    }                                                                          \
  }

// TODO add doxygen for individual overloads.
#define __ESIMD_BITWISE_OP_FILTER                                              \
  std::is_integral_v<T1> &&std::is_integral_v<T2>
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(^, BinOp::bit_xor, __ESIMD_BITWISE_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(|, BinOp::bit_or, __ESIMD_BITWISE_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(&, BinOp::bit_and, __ESIMD_BITWISE_OP_FILTER)
#undef __ESIMD_BITWISE_OP_FILTER

#define __ESIMD_SHIFT_OP_FILTER                                                \
  std::is_integral_v<T1> &&std::is_integral_v<T2>                              \
      &&__SEIEED::is_simd_type_v<SimdTx>
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(%, BinOp::rem, __ESIMD_SHIFT_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(<<, BinOp::shl, __ESIMD_SHIFT_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(>>, BinOp::shr, __ESIMD_SHIFT_OP_FILTER)
#undef __ESIMD_SHIFT_OP_FILTER

#define __ESIMD_ARITH_OP_FILTER                                                \
  __SEIEED::is_valid_simd_elem_type_v<T1>                                      \
      &&__SEIEED::is_valid_simd_elem_type_v<T2>                                \
          &&__SEIEED::is_simd_type_v<SimdTx>

__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(+, BinOp::add, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(-, BinOp::sub, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(*, BinOp::mul, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP(/, BinOp::div, __ESIMD_ARITH_OP_FILTER)
#undef __ESIMD_ARITH_OP_FILTER

#undef __ESIMD_DEF_SIMD_OBJ_IMPL_BIN_OP

// ========= simd_obj_impl comparison operators
// Both simd and simd_mask will match simd_obj_impl argument when resolving
// operator overloads.

#define __ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(CMPOP, CMPOP_ID, COND)                \
                                                                               \
  /* simd_obj_impl CMPOP simd_obj_impl */                                      \
  template <class T1, class T2, int N, template <class, int> class SimdT,      \
            class SimdTx = SimdT<T1, N>, class = std::enable_if_t<COND>>       \
  inline __SEIEE::simd_mask<N> operator CMPOP(                                 \
      const __SEIEED::simd_obj_impl<__raw_t<T1>, N, SimdT<T1, N>> &LHS,        \
      const __SEIEED::simd_obj_impl<__raw_t<T2>, N, SimdT<T2, N>> &RHS) {      \
    using MaskVecT = typename __SEIEE::simd_mask<N>::raw_vector_type;          \
                                                                               \
    if constexpr (is_simd_type_v<SimdT<T1, N>>) {                              \
      using PromotedT = computation_type_t<T1, T2>;                            \
      /* vector_comparison_op returns vector_type_t<Ti, N>, where Ti is        \
       * integer type */                                                       \
      /* of the same bit size as PromotedT */                                  \
      auto Res = vector_comparison_op<CMPOP_ID, PromotedT, N>(                 \
          __SEIEED::convert_vector<PromotedT, T1, N>(LHS.data()),              \
          __SEIEED::convert_vector<PromotedT, T2, N>(RHS.data()));             \
      using ResElemT = element_type_t<decltype(Res)>;                          \
      return __SEIEE::simd_mask<N>(                                            \
          __SEIEED::convert_vector<simd_mask_elem_type, ResElemT, N>(Res) &    \
          MaskVecT(1));                                                        \
    } else {                                                                   \
      /* this is comparison of masks, don't perform type promotion */          \
      auto ResVec = LHS.data() CMPOP RHS.data();                               \
      return __SEIEE::simd_mask<N>(__SEIEED::convert<MaskVecT>(ResVec) &       \
                                   MaskVecT(1));                               \
    }                                                                          \
  }                                                                            \
                                                                               \
  /* simd_obj_impl CMPOP SCALAR */                                             \
  template <class T1, int N1, template <class, int> class SimdT1, class T2,    \
            class SimdTx = SimdT1<T1, N1>,                                     \
            class = std::enable_if_t<                                          \
                __SEIEED::is_valid_simd_elem_type_v<T2> && COND>>              \
  inline __SEIEE::simd_mask<N1> operator CMPOP(                                \
      const __SEIEED::simd_obj_impl<__raw_t<T1>, N1, SimdT1<T1, N1>> &LHS,     \
      T2 RHS) {                                                                \
    if constexpr (__SEIEED::is_simd_type_v<SimdT1<T1, N1>>)                    \
      /* simd case */                                                          \
      return LHS CMPOP SimdT1<T2, N1>(RHS);                                    \
    else                                                                       \
      /* simd_mask case - element type is fixed */                             \
      return LHS CMPOP SimdT1<T1, N1>(convert_scalar<T1>(RHS));                \
  }                                                                            \
                                                                               \
  /* SCALAR CMPOP simd_obj_impl */                                             \
  template <class T1, class T2, int N2, template <class, int> class SimdT2,    \
            class SimdTx = SimdT2<T2, N2>,                                     \
            class = std::enable_if_t<                                          \
                __SEIEED::is_valid_simd_elem_type_v<T1> && COND>>              \
  inline __SEIEE::simd_mask<N2> operator CMPOP(                                \
      T1 LHS,                                                                  \
      const __SEIEED::simd_obj_impl<__raw_t<T2>, N2, SimdT2<T2, N2>> &RHS) {   \
    if constexpr (__SEIEED::is_simd_type_v<SimdT2<T2, N2>>)                    \
      /* simd case */                                                          \
      return SimdT2<T1, N2>(LHS) CMPOP RHS;                                    \
    else                                                                       \
      /* simd_mask case - element type is fixed */                             \
      return SimdT2<T2, N2>(convert_scalar<T2>(LHS)) CMPOP RHS;                \
  }

// Equality comparison is defined for all simd_obj_impl subclasses.
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(==, CmpOp::eq, true)
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(!=, CmpOp::ne, true)

// Relational operators are defined only for the simd type.
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(<, CmpOp::lt, __SEIEED::is_simd_type_v<SimdTx>)
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(>, CmpOp::gt, __SEIEED::is_simd_type_v<SimdTx>)
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(<=, CmpOp::lte,
                                 __SEIEED::is_simd_type_v<SimdTx>)
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(>=, CmpOp::gte,
                                 __SEIEED::is_simd_type_v<SimdTx>)

// Logical operators are defined only for the simd_mask type
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(&&, BinOp::log_and,
                                 __SEIEED::is_simd_mask_type_v<SimdTx>)
__ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP(||, BinOp::log_or,
                                 __SEIEED::is_simd_mask_type_v<SimdTx>)

#undef __ESIMD_DEF_SIMD_OBJ_IMPL_CMP_OP

////////////////////////////////////////////////////////////////////////////////
// simd_view global operators
////////////////////////////////////////////////////////////////////////////////

// ========= simd_view bitwise logic and arithmetic operators

#define __ESIMD_DEF_SIMD_VIEW_BIN_OP(BINOP, COND)                              \
                                                                               \
  /* simd_view BINOP simd_view */                                              \
  template <class SimdT1, class RegionT1, class SimdT2, class RegionT2,        \
            class T1 = typename __SEIEE::shape_type<RegionT1>::element_type,   \
            class T2 = typename __SEIEE::shape_type<RegionT2>::element_type,   \
            auto N1 = __SEIEE::shape_type<RegionT1>::length,                   \
            auto N2 = __SEIEE::shape_type<RegionT2>::length,                   \
            class =                                                            \
                std::enable_if_t<__SEIEED::is_simd_type_v<SimdT1> ==           \
                                     __SEIEED::is_simd_type_v<SimdT2> &&       \
                                 (N1 == N2 || N1 == 1 || N2 == 1) && COND>>    \
  inline auto operator BINOP(                                                  \
      const __SEIEE::simd_view<SimdT1, RegionT1> &LHS,                         \
      const __SEIEE::simd_view<SimdT2, RegionT2> &RHS) {                       \
    if constexpr (N1 == 1)                                                     \
      return (T1)LHS.read()[0] BINOP RHS.read();                               \
    else if constexpr (N2 == 1)                                                \
      return LHS.read() BINOP(T2) RHS.read()[0];                               \
    else                                                                       \
      return LHS.read() BINOP RHS.read();                                      \
  }                                                                            \
                                                                               \
  /* simd* BINOP simd_view<simd*...> */                                        \
  template <class SimdT1, class SimdT2, class RegionT2,                        \
            class T1 = typename SimdT1::element_type,                          \
            class T2 = typename __SEIEE::shape_type<RegionT2>::element_type,   \
            class = std::enable_if_t<                                          \
                __SEIEED::is_simd_obj_impl_derivative_v<SimdT1> &&             \
                (__SEIEED::is_simd_type_v<SimdT1> ==                           \
                 __SEIEED::is_simd_type_v<SimdT2>)&&(SimdT1::length ==         \
                                                     __SEIEE::shape_type<      \
                                                         RegionT2>::length) && \
                COND>>                                                         \
  inline auto operator BINOP(                                                  \
      const SimdT1 &LHS, const __SEIEE::simd_view<SimdT2, RegionT2> &RHS) {    \
    return LHS BINOP RHS.read();                                               \
  }                                                                            \
                                                                               \
  /* simd_view<simd*...> BINOP simd* */                                        \
  template <                                                                   \
      class SimdT1, class RegionT1, class SimdT2,                              \
      class T1 = typename __SEIEE::shape_type<RegionT1>::element_type,         \
      class T2 = typename SimdT2::element_type,                                \
      class = std::enable_if_t<                                                \
          __SEIEED::is_simd_obj_impl_derivative_v<SimdT2> &&                   \
          __SEIEED::is_simd_type_v<SimdT1> ==                                  \
              __SEIEED::is_simd_type_v<SimdT2> &&                              \
          (SimdT2::length == __SEIEE::shape_type<RegionT1>::length) && COND>>  \
  inline auto operator BINOP(const __SEIEE::simd_view<SimdT1, RegionT1> &LHS,  \
                             const SimdT2 &RHS) {                              \
    return LHS.read() BINOP RHS;                                               \
  }                                                                            \
                                                                               \
  /* SCALAR BINOP simd_view */                                                 \
  template <class T1, class SimdViewT2,                                        \
            class T2 = typename SimdViewT2::element_type,                      \
            class SimdT1 = typename SimdViewT2::value_type,                    \
            class = std::enable_if_t<                                          \
                __SEIEED::is_any_simd_view_type_v<SimdViewT2> && COND>>        \
  inline auto operator BINOP(T1 LHS, const SimdViewT2 &RHS) {                  \
    return LHS BINOP RHS.read();                                               \
  }                                                                            \
                                                                               \
  /* simd_view BINOP SCALAR */                                                 \
  template <class SimdViewT1, class T2,                                        \
            class T1 = typename SimdViewT1::element_type,                      \
            class SimdT1 = typename SimdViewT1::value_type,                    \
            class = std::enable_if_t<                                          \
                __SEIEED::is_any_simd_view_type_v<SimdViewT1> && COND>>        \
  inline auto operator BINOP(const SimdViewT1 &LHS, T2 RHS) {                  \
    return LHS.read() BINOP RHS;                                               \
  }

#define __ESIMD_BITWISE_OP_FILTER                                              \
  std::is_integral_v<T1> &&std::is_integral_v<T2>
__ESIMD_DEF_SIMD_VIEW_BIN_OP(^, __ESIMD_BITWISE_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(|, __ESIMD_BITWISE_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(&, __ESIMD_BITWISE_OP_FILTER)
#undef __ESIMD_BITWISE_OP_FILITER

#define __ESIMD_SHIFT_OP_FILTER                                                \
  std::is_integral_v<T1> &&std::is_integral_v<T2>                              \
      &&__SEIEED::is_simd_type_v<SimdT1>

__ESIMD_DEF_SIMD_VIEW_BIN_OP(%, __ESIMD_SHIFT_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(<<, __ESIMD_SHIFT_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(>>, __ESIMD_SHIFT_OP_FILTER)
#undef __ESIMD_SHIFT_OP_FILTER

#define __ESIMD_ARITH_OP_FILTER                                                \
  __SEIEED::is_simd_type_v<SimdT1> &&__SEIEED::is_valid_simd_elem_type_v<T1>   \
      &&__SEIEED::is_valid_simd_elem_type_v<T2>

__ESIMD_DEF_SIMD_VIEW_BIN_OP(+, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(-, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(*, __ESIMD_ARITH_OP_FILTER)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(/, __ESIMD_ARITH_OP_FILTER)
#undef __ESIMD_ARITH_OP_FILTER

__ESIMD_DEF_SIMD_VIEW_BIN_OP(&&, __SEIEED::is_simd_mask_type_v<SimdT1>)
__ESIMD_DEF_SIMD_VIEW_BIN_OP(||, __SEIEED::is_simd_mask_type_v<SimdT1>)

#undef __ESIMD_DEF_SIMD_VIEW_BIN_OP

// ========= simd_view comparison operators

#define __ESIMD_DEF_SIMD_VIEW_CMP_OP(CMPOP, COND)                              \
                                                                               \
  /* simd_view CMPOP simd_view */                                              \
  template <class SimdT1, class RegionT1, class SimdT2, class RegionT2,        \
            auto N1 = __SEIEE::shape_type<RegionT1>::length,                   \
            auto N2 = __SEIEE::shape_type<RegionT2>::length,                   \
            class = std::enable_if_t</* both views must have the same base     \
                                        type kind - simds or masks: */         \
                                     (__SEIEED::is_simd_type_v<SimdT1> ==      \
                                      __SEIEED::is_simd_type_v<                \
                                          SimdT2>)&&/* the length of the views \
                                                       must match as well: */  \
                                     (N1 == N2 || N1 == 1 || N2 == 1) &&       \
                                     COND>>                                    \
  inline auto operator CMPOP(                                                  \
      const __SEIEE::simd_view<SimdT1, RegionT1> &LHS,                         \
      const __SEIEE::simd_view<SimdT2, RegionT2> &RHS) {                       \
    using T1 = typename __SEIEE::shape_type<RegionT1>::element_type;           \
    using T2 = typename __SEIEE::shape_type<RegionT2>::element_type;           \
    if constexpr (N1 == 1)                                                     \
      return (T1)LHS.read()[0] CMPOP RHS.read();                               \
    else if constexpr (N2 == 1)                                                \
      return LHS.read() CMPOP(T2) RHS.read()[0];                               \
    else                                                                       \
      return LHS.read() CMPOP RHS.read();                                      \
  }                                                                            \
                                                                               \
  /* simd_view CMPOP simd_obj_impl */                                          \
  template <class SimdT1, class RegionT1, class RawT2, int N2, class SimdT2,   \
            class = std::enable_if_t<                                          \
                (__SEIEE::shape_type<RegionT1>::length == N2) &&               \
                (__SEIEED::is_simd_type_v<SimdT1> ==                           \
                 __SEIEED::is_simd_type_v<SimdT2>)&&COND>>                     \
  inline __SEIEE::simd_mask<N2> operator CMPOP(                                \
      const __SEIEE::simd_view<SimdT1, RegionT1> &LHS,                         \
      const __SEIEED::simd_obj_impl<RawT2, N2, SimdT2> &RHS) {                 \
    return LHS.read() CMPOP SimdT2(RHS.data());                                \
  }                                                                            \
                                                                               \
  /* simd_obj_impl CMPOP simd_view */                                          \
  template <class RawT1, int N1, class SimdT1, class SimdT2, class RegionT2,   \
            class = std::enable_if_t<                                          \
                (__SEIEE::shape_type<RegionT2>::length == N1) &&               \
                (__SEIEED::is_simd_type_v<SimdT1> ==                           \
                 __SEIEED::is_simd_type_v<SimdT2>)&&COND>>                     \
  inline __SEIEE::simd_mask<N1> operator CMPOP(                                \
      const __SEIEED::simd_obj_impl<RawT1, N1, SimdT1> &LHS,                   \
      const __SEIEE::simd_view<SimdT2, RegionT2> &RHS) {                       \
    return SimdT1(LHS.data()) CMPOP RHS.read();                                \
  }                                                                            \
                                                                               \
  /* simd_view CMPOP SCALAR */                                                 \
  template <class SimdT1, class RegionT1, class T2,                            \
            class = std::enable_if_t<                                          \
                __SEIEED::is_valid_simd_elem_type_v<T2> && COND>>              \
  inline auto operator CMPOP(const __SEIEE::simd_view<SimdT1, RegionT1> &LHS,  \
                             T2 RHS) {                                         \
    return LHS.read() CMPOP RHS;                                               \
  }                                                                            \
                                                                               \
  /* SCALAR CMPOP simd_view */                                                 \
  template <class T1, class SimdT2, class RegionT2, class SimdT1 = SimdT2,     \
            class = std::enable_if_t<                                          \
                __SEIEED::is_valid_simd_elem_type_v<T1> && COND>>              \
  inline auto operator CMPOP(                                                  \
      T1 LHS, const __SEIEE::simd_view<SimdT2, RegionT2> &RHS) {               \
    return LHS CMPOP RHS.read();                                               \
  }

// Equality comparison is defined for views of all simd_obj_impl derivatives.
__ESIMD_DEF_SIMD_VIEW_CMP_OP(==, true)
__ESIMD_DEF_SIMD_VIEW_CMP_OP(!=, true)

// Relational operators are defined only for views of the simd class.
__ESIMD_DEF_SIMD_VIEW_CMP_OP(<, __SEIEED::is_simd_type_v<SimdT1>)
__ESIMD_DEF_SIMD_VIEW_CMP_OP(>, __SEIEED::is_simd_type_v<SimdT1>)
__ESIMD_DEF_SIMD_VIEW_CMP_OP(<=, __SEIEED::is_simd_type_v<SimdT1>)
__ESIMD_DEF_SIMD_VIEW_CMP_OP(>=, __SEIEED::is_simd_type_v<SimdT1>)

#undef __ESIMD_DEF_SIMD_VIEW_CMP_OP

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
