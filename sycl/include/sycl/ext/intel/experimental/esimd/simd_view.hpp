//==------------ - simd_view.hpp - DPC++ Explicit SIMD API   ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD vector view APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/simd_view_impl.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

/// @addtogroup sycl_esimd_core_vectors
/// @{

/// This class represents a reference to a sub-region of a base simd object.
/// The referenced sub-region of the base object can be read from and written to
/// via an instance of this class. Derives from detail::simd_view_impl, which
/// defines the majority of available APIs for view classes.
///
/// User code is never supposed to explicitly provide actual template parameters
/// for this class. They are always auto-deduced or provided by APIs.
/// @tparam BaseTy The base type - type of the object viewed by this one.
/// @tparam RegionTy Describes the viewed region - its shape and element type.
///   Regions can be 1D and 2D.
///
template <typename BaseTy,
          typename RegionTy =
              region1d_t<typename BaseTy::element_type, BaseTy::length, 1>>
class simd_view : public detail::simd_view_impl<BaseTy, RegionTy> {
  /// @cond ESIMD_DETAIL
  template <typename, int, class, class> friend class detail::simd_obj_impl;
  template <typename, int> friend class detail::simd_mask_impl;
  template <typename, typename> friend class simd_view;
  template <typename, int> friend class simd;
  template <typename, typename> friend class detail::simd_view_impl;

protected:
  using BaseClass = detail::simd_view_impl<BaseTy, RegionTy>;
  // Deduce the corresponding value type from its region type.
  using ShapeTy = typename shape_type<RegionTy>::type;
  using base_type = BaseTy;
  template <typename ElT, int N>
  using get_simd_t = typename BaseClass::template get_simd_t<ElT, N>;
  /// @endcond ESIMD_DETAIL

public:
  static_assert(detail::is_simd_obj_impl_derivative_v<BaseTy>);

  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  /// The region type of this class.
  using region_type = RegionTy;

  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = typename ShapeTy::element_type;

  /// The simd type if reading the object.
  using value_type = get_simd_t<element_type, length>;

  /// The underlying builtin value type
  using raw_vector_type =
      detail::vector_type_t<detail::__raw_t<element_type>, length>;

protected:
  /// @cond ESIMD_DETAIL
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}
  /// @endcond ESIMD_DETAIL

public:
  /// Default copy and move constructors for simd_view.
  simd_view(const simd_view &Other) = default;
  simd_view(simd_view &&Other) = default;

  /// Construct a complete view of a vector
  /// @param Base The vector to construct a view of.
  simd_view(BaseTy &Base) : BaseClass(Base) {}

  /// Copy assignment operator.
  simd_view &operator=(const simd_view &Other) {
    BaseClass::operator=(Other);
    return *this;
  }

  using BaseClass::operator--;
  using BaseClass::operator++;
  using BaseClass::operator=;
};

#define __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(RELOP)                              \
  /* simd_view RELOP simd_view */                                              \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X,                  \
                                          const simd_view &Y) {                \
    return (element_type)X RELOP(element_type) Y;                              \
  }                                                                            \
                                                                               \
  /* simd_view RELOP SCALAR */                                                 \
  template <typename T1,                                                       \
            std::enable_if_t<detail::is_valid_simd_elem_type_v<T1>>>           \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X, T1 Y) {          \
    return (element_type)X RELOP Y;                                            \
  }                                                                            \
                                                                               \
  /* SCALAR RELOP simd_view */                                                 \
  template <typename T1,                                                       \
            std::enable_if_t<detail::is_valid_simd_elem_type_v<T1>>>           \
  ESIMD_INLINE friend bool operator RELOP(T1 X, const simd_view &Y) {          \
    return X RELOP(element_type) Y;                                            \
  }

/// This is a specialization of simd_view class with a single element.
/// Objects of such a class are created in the following situation:
///   simd<int, 4> v = 1;
///   auto v1 = v[0];
/// We allow implicit conversion to underlying type, e.g.:
///   simd<int, 4> v = 1;
///   int i = v[0];
/// Also, relational operators with such objects return a scalar bool value
/// instead of a mask, to allow:
///   bool b = v[0] > v[1] && v[2] < 42;
///
template <typename BaseTy, class ViewedElemT>
class simd_view<BaseTy, region1d_scalar_t<ViewedElemT>>
    : public detail::simd_view_impl<BaseTy, region1d_scalar_t<ViewedElemT>> {
  template <typename, int, class, class> friend class detail::simd_obj_impl;
  template <typename, typename> friend class detail::simd_view_impl;

public:
  using RegionTy = region1d_scalar_t<ViewedElemT>;
  using BaseClass = detail::simd_view_impl<BaseTy, RegionTy>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;
  static_assert(1 == length, "length of this view is not equal to 1");
  static_assert(std::is_same_v<typename ShapeTy::element_type, ViewedElemT>);
  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = ViewedElemT;
  using base_type = BaseTy;
  template <typename ElT, int N>
  using get_simd_t = typename BaseClass::template get_simd_t<ElT, N>;
  /// The simd type if reading the object.
  using value_type = get_simd_t<element_type, length>;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  /// Construct a complete view of a vector
  simd_view(BaseTy &Base) : BaseClass(Base) {}

  operator element_type() const {
    const auto v = BaseClass::read().data();
    return detail::bitcast_to_wrapper_type<element_type>(std::move(v)[0]);
  }

  using BaseClass::operator--;
  using BaseClass::operator++;
  using BaseClass::operator=;

  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(>)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(>=)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(<)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(<=)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(==)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(!=)
};

// TODO: remove code duplication in two class specializations for a simd_view
// with a single element

/// This is a specialization of nested simd_view class with a single element.
/// Objects of such a class are created in the following situation:
///   simd<int, 4> v = 1;
///   auto v1 = v.select<2, 1>(0);
///   auto v2 = v1[0]; // simd_view of a nested region for a single element
template <typename BaseTy, typename NestedRegion, class ViewedElemT>
class simd_view<BaseTy, std::pair<region1d_scalar_t<ViewedElemT>, NestedRegion>>
    : public detail::simd_view_impl<
          BaseTy, std::pair<region1d_scalar_t<ViewedElemT>, NestedRegion>> {
  template <typename, int> friend class simd;
  template <typename, typename> friend class detail::simd_view_impl;

public:
  using RegionTy = std::pair<region1d_scalar_t<ViewedElemT>, NestedRegion>;
  using BaseClass = detail::simd_view_impl<BaseTy, RegionTy>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;
  static_assert(1 == length, "length of this view is not equal to 1");
  static_assert(std::is_same_v<typename ShapeTy::element_type, ViewedElemT>);
  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = ViewedElemT;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  using BaseClass::operator=;

  operator element_type() const {
    const auto v = BaseClass::read();
    return detail::convert_scalar<element_type>(v[0]);
  }

  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(>)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(>=)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(<)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(<=)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(==)
  __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP(!=)
};

#undef __ESIMD_DEF_SCALAR_SIMD_VIEW_RELOP

/// @} sycl_esimd_core_vectors

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
