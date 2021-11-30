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

#include <sycl/ext/intel/experimental/esimd/detail/simd_mask_impl.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/simd_obj_impl.hpp>

#include <sycl/ext/intel/experimental/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <iostream>
#endif // __SYCL_DEVICE_ONLY__

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

/// The simd vector class.
///
/// This is a wrapper class for llvm vector values. Additionally this class
/// supports region operations that map to Intel GPU regions. The type of
/// a region select or format operation is of simd_view type, which models
/// read-update-write semantics.
///
/// \ingroup sycl_esimd
template <typename Ty, int N>
class simd
    : public detail::simd_obj_impl<
          Ty, N, simd<Ty, N>, std::enable_if_t<detail::is_vectorizable_v<Ty>>> {
  using base_type = detail::simd_obj_impl<Ty, N, simd<Ty, N>>;

public:
  using base_type::base_type;
  using element_type = typename base_type::element_type;
  using vector_type = typename base_type::vector_type;
  static constexpr int length = N;

  // Implicit conversion constructor from another simd object of the same
  // length.
  template <typename SimdT,
            class = std::enable_if_t<__SEIEED::is_simd_type_v<SimdT> &&
                                     (length == SimdT::length)>>
  simd(const SimdT &RHS)
      : base_type(__builtin_convertvector(RHS.data(), vector_type)) {
    __esimd_dbg_print(simd(const SimdT &RHS));
  }

  // Broadcast constructor with conversion.
  template <typename T1,
            class = std::enable_if_t<detail::is_vectorizable_v<T1>>>
  simd(T1 Val) : base_type((Ty)Val) {
    __esimd_dbg_print(simd(T1 Val));
  }

  /// Explicit conversion for simd_obj_impl<T, 1> into T.
  template <class To, class T = simd,
            class = sycl::detail::enable_if_t<(T::length == 1) &&
                                              detail::is_vectorizable_v<To>>>
  operator To() const {
    __esimd_dbg_print(explicit operator To());
    return (To)base_type::data()[0];
  }

  /// @{
  /// Infix and postfix operators ++, --
  simd &operator++() {
    *this += 1;
    return *this;
  }

  simd operator++(int) {
    simd Ret(base_type::data());
    operator++();
    return Ret;
  }

  simd &operator--() {
    *this -= 1;
    return *this;
  }

  simd operator--(int) {
    simd Ret(base_type::data());
    operator--();
    return Ret;
  }
  /// @}

#define __ESIMD_DEF_SIMD_ARITH_UNARY_OP(ARITH_UNARY_OP)                        \
  template <class T1 = Ty> simd operator ARITH_UNARY_OP() const {              \
    static_assert(!std::is_unsigned_v<T1>,                                     \
                  #ARITH_UNARY_OP "doesn't apply to unsigned types");          \
    return simd(ARITH_UNARY_OP(base_type::data()));                            \
  }

  __ESIMD_DEF_SIMD_ARITH_UNARY_OP(-)
  __ESIMD_DEF_SIMD_ARITH_UNARY_OP(+)
#undef __ESIMD_DEF_SIMD_ARITH_UNARY_OP
};

/// Covert from a simd object with element type \c From to a simd object with
/// element type \c To.
template <typename To, typename From, int N>
ESIMD_INLINE simd<To, N> convert(const simd<From, N> &val) {
  if constexpr (std::is_same_v<To, From>)
    return val;
  else
    return __builtin_convertvector(val.data(), detail::vector_type_t<To, N>);
}

#undef __ESIMD_DEF_RELOP
#undef __ESIMD_DEF_BITWISE_OP

/// Represents a simd mask.
template <int N> using simd_mask = detail::simd_mask_type<N>;

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

template <typename Ty, int N>
std::ostream &operator<<(std::ostream &OS, const __SEIEE::simd<Ty, N> &V)
#ifdef __SYCL_DEVICE_ONLY__
    ;
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
