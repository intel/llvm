//==------------ - simd_mask_impl.hpp - DPC++ Explicit SIMD API   ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation detail of Explicit SIMD mask class.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/simd_obj_impl.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

/// @addtogroup sycl_esimd_core_vectors
/// @{

/// This class is a simd_obj_impl specialization representing a simd mask, which
/// is basically a simd_obj_impl with fixed element type (uint16_t, not publicly
/// usable) and limited set of APIs. E.g. arithmetic operations (\c +, \c -,
/// etc.) are not defined for masks, but bit operations like \c ^, \c & are.
/// @tparam T Fixed element type, must be simd_mask_elem_type.
/// @tparam N Number of elements (per-lane predicates) in the mask.
template <typename T, int N>
class simd_mask_impl
    : public simd_obj_impl<
          T, N, simd_mask_impl<T, N>,
          std::enable_if_t<std::is_same_v<simd_mask_elem_type, T>>> {
  /// @cond ESIMD_DETAIL
  using base_type = simd_obj_impl<T, N, simd_mask_impl<T, N>>;
  /// @endcond ESIMD_DETAIL

public:
  /// Raw element type actually used for storage.
  using raw_element_type = T;
  /// Element type, same as raw.
  using element_type = T;
  /// Underlying storage type for the entire vector.
  using raw_vector_type = typename base_type::raw_vector_type;
  static_assert(std::is_same_v<raw_vector_type, simd_mask_storage_t<N>> &&
                "mask impl type mismatch");

  /// Compiler-generated default constructor.
  simd_mask_impl() = default;

  /// Copy constructor.
  simd_mask_impl(const simd_mask_impl &other) : base_type(other) {}

  /// Broadcast constructor with conversion. @see
  /// simd_obj_impl::simd_obj_impl(T)
  template <class T1, class = std::enable_if_t<std::is_integral_v<T1>>>
  simd_mask_impl(T1 Val) : base_type((T)Val) {}

  /// Implicit conversion constructor from a raw vector object.
  // TODO this should be made inaccessible from user code.
  simd_mask_impl(const raw_vector_type &Val) : base_type(Val) {}

  /// Construct from an array. To allow e.g. simd_mask<N> m({1,0,0,1,...}).
  template <int N1, class = std::enable_if_t<N1 == N>>
  simd_mask_impl(const raw_element_type (&&Arr)[N1]) {
    base_type::template init_from_array<false>(std::move(Arr));
  }

  /// Implicit conversion from simd.
  simd_mask_impl(const simd<T, N> &Val) : base_type(Val.data()) {}

private:
  /// @cond ESIMD_DETAIL
  static inline constexpr bool mask_size_ok_for_mem_io() {
    constexpr unsigned Sz = sizeof(element_type) * N;
    return (Sz >= OperandSize::OWORD) && (Sz % OperandSize::OWORD == 0) &&
           isPowerOf2(Sz / OperandSize::OWORD) &&
           (Sz <= 8 * OperandSize::OWORD);
  }
  /// @endcond ESIMD_DETAIL

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

  /// Conversion to boolean. Available only when the number of elements is 1.
  /// @return true if the element is non-zero, false otherwise.
  template <class T1 = simd_mask_impl,
            class = std::enable_if_t<T1::length == 1>>
  operator bool() const {
    return base_type::data()[0] != 0;
  }
};

#undef __ESIMD_MASK_DEPRECATION_MSG

/// @} sycl_esimd_core_vectors

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
