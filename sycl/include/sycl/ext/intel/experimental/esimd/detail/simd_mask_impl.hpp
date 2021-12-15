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

#define __ESIMD_MASK_DEPRECATION_MSG                                           \
  "Use of 'simd'/'simd_view<simd,...>' class to represent predicate or mask "  \
  "is deprecated. Use "                                                        \
  "'simd_mask'/'simd_view<simd_mask,...>' instead."

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

  /// Implicit conversion from simd_view<simd,...>.
  template <
      // viewed simd class parameters
      int N1, class T1,
      // view region
      class RegionT2,
      // view element type
      class T2 = typename __SEIEE::shape_type<RegionT2>::element_type,
      // view size in elements
      int N2 = __SEIEE::shape_type<RegionT2>::length,
      // enable only if view length and element type match this object
      class = std::enable_if_t<N == N2 && std::is_same_v<T, T2>>>
  __SYCL_DEPRECATED(__ESIMD_MASK_DEPRECATION_MSG)
  simd_mask_impl(const simd_view<simd<T1, N1>, RegionT2> &Val)
      : base_type(Val.read().data()) {}

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
  operator bool() const {
    return base_type::data()[0] != 0;
  }
};

#undef __ESIMD_MASK_DEPRECATION_MSG

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
