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

#include <sycl/ext/intel/experimental/esimd/detail/simd_obj_impl.hpp>

#include <sycl/ext/intel/experimental/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <iostream>
#endif // __SYCL_DEVICE_ONLY__

#define __ESIMD_MASK_DEPRECATION_MSG                                           \
  "Use of 'simd' class to represent predicate or mask is deprecated. Use "     \
  "'simd_mask' instead."

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
  template <class T1 = Ty> simd operator ARITH_UNARY_OP() {                    \
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

namespace detail {
template <typename T, int N>
class simd_mask_impl
    : public detail::simd_obj_impl<
          T, N, simd_mask_impl<T, N>,
          std::enable_if_t<std::is_same_v<detail::simd_mask_elem_type, T>>> {
  using base_type = detail::simd_obj_impl<T, N, simd_mask_impl<T, N>>;

public:
  using element_type = T;
  using vector_type = typename base_type::vector_type;
  static_assert(std::is_same_v<vector_type, simd_mask_storage_t<N>> &&
                "mask impl type mismatch");

  simd_mask_impl() = default;
  simd_mask_impl(const simd_mask_impl &other) : base_type(other) {}

  /// Broadcast constructor with conversion.
  template <class T1, class = std::enable_if_t<std::is_integral_v<T1>>>
  simd_mask_impl(T1 Val) : base_type((T)Val) {}

  /// Implicit conversion constructor from a raw vector object.
  // TODO this should be made inaccessible from user code.
  simd_mask_impl(const vector_type &Val) : base_type(Val) {}

  /// Initializer list constructor.
  __SYCL_DEPRECATED("use constructor from array, e.g: simd_mask<3> x({0,1,1});")
  simd_mask_impl(std::initializer_list<T> Ilist) : base_type(Ilist) {}

  /// Construct from an array. To allow e.g. simd_mask<N> m({1,0,0,1,...}).
  template <int N1, class = std::enable_if_t<N1 == N>>
  simd_mask_impl(const element_type(&&Arr)[N1]) {
    base_type::template init_from_array<N1>(std::move(Arr));
  }

  /// Implicit conversion from simd.
  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask_impl(const simd<T, N> &Val) : base_type(Val.data()) {}

private:
  static inline constexpr bool mask_size_ok_for_mem_io() {
    constexpr unsigned Sz = sizeof(element_type) * N;
    return (Sz >= detail::OperandSize::OWORD) &&
           (Sz % detail::OperandSize::OWORD == 0) &&
           detail::isPowerOf2(Sz / detail::OperandSize::OWORD) &&
           (Sz <= 8 * detail::OperandSize::OWORD);
  }

public:
  // TODO add accessor-based mask memory operations.

  /// Load constructor.
  // Implementation note: use SFINAE to avoid overload ambiguity:
  // 1) with 'simd_mask(element_type v)' in 'simd_mask<N> m(0)'
  // 2) with 'simd_mask(const T1(&&arr)[N])' in simd_mask<N>
  // m((element_type*)p)'
  template <typename T1,
            typename = std::enable_if_t<mask_size_ok_for_mem_io() &&
                                        std::is_same_v<T1, element_type>>>
  explicit simd_mask_impl(const T1 *ptr) {
    base_type::copy_from(ptr);
  }

  /// Broadcast assignment operator to support simd_mask_impl<N> n = a > b;
  simd_mask_impl &operator=(element_type val) noexcept {
    base_type::set(val);
    return *this;
  }

  template <class T1 = simd_mask_impl,
            class = std::enable_if_t<T1::length == 1>>
  operator bool() {
    return base_type::data()[0] != 0;
  }
};

} // namespace detail

#undef __ESIMD_DEF_RELOP
#undef __ESIMD_DEF_BITWISE_OP

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
