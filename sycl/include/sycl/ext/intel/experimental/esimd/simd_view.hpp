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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

/// This class represents a reference to a sub-region of a base simd object.
/// The referenced sub-region of the base object can be read from and written to
/// via an instance of this class.
///
/// \ingroup sycl_esimd
template <typename BaseTy, typename RegionTy>
class simd_view : public detail::simd_view_impl<BaseTy, RegionTy,
                                                simd_view<BaseTy, RegionTy>> {
  template <typename, int> friend class simd;
  template <typename, typename, typename> friend class detail::simd_view_impl;

public:
  using BaseClass =
      detail::simd_view_impl<BaseTy, RegionTy, simd_view<BaseTy, RegionTy>>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  /// The simd type if reading this simd_view object.
  using value_type = simd<typename ShapeTy::element_type, length>;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  // Default copy and move constructors for simd_view.
  simd_view(const simd_view &Other) = default;
  simd_view(simd_view &&Other) = default;

  /// @{
  /// Assignment operators.
  simd_view &operator=(const simd_view &Other) {
    *this = Other.read();
    return *this;
  }
  simd_view &operator=(const value_type &Val) {
    this->M_base.writeRegion(this->M_region, Val.data());
    return *this;
  }
  /// @}

  /// Move assignment operator.
  simd_view &operator=(simd_view &&Other) {
    *this = Other.read();
    return *this;
  }

#define DEF_RELOP(RELOP)                                                       \
  ESIMD_INLINE friend simd<uint16_t, length> operator RELOP(                   \
      const simd_view &X, const value_type &Y) {                               \
    auto R = X.read().data() RELOP Y.data();                                   \
    mask_type_t<length> M(1);                                                  \
    return M & detail::convert<mask_type_t<length>>(R);                        \
  }                                                                            \
  ESIMD_INLINE friend simd<uint16_t, length> operator RELOP(                   \
      const value_type &X, const simd_view &Y) {                               \
    auto R = X.data() RELOP Y.read().data();                                   \
    mask_type_t<length> M(1);                                                  \
    return M & detail::convert<mask_type_t<length>>(R);                        \
  }                                                                            \
  ESIMD_INLINE friend simd<uint16_t, length> operator RELOP(                   \
      const simd_view &X, const simd_view &Y) {                                \
    return (X RELOP Y.read());                                                 \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP
};

/// This is a specialization of simd_view class with a single element.
/// We allow implicit conversion to underlying type, e.g.:
///   simd<int, 4> v = 1;
///   int i = v[0];
/// Also, relational operators with such objects return a scalar bool value
/// instead of a mask, to allow:
///   bool b = v[0] > v[1] && v[2] < 42;
///
/// \ingroup sycl_esimd
template <typename BaseTy>
class simd_view<BaseTy, region_base_1<typename BaseTy::element_type>>
    : public detail::simd_view_impl<
          BaseTy, region_base_1<typename BaseTy::element_type>,
          simd_view<BaseTy, region_base_1<typename BaseTy::element_type>>> {
  template <typename, int> friend class simd;
  template <typename, typename, typename> friend class detail::simd_view_impl;

public:
  using RegionTy = region_base_1<typename BaseTy::element_type>;
  using BaseClass =
      detail::simd_view_impl<BaseTy, RegionTy, simd_view<BaseTy, RegionTy>>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;
  static_assert(1 == length, "length of this view is not equal to 1");
  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = typename ShapeTy::element_type;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  operator element_type() const { return (*this)[0]; }

  using BaseClass::operator=;

#define DEF_RELOP(RELOP)                                                       \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X,                  \
                                          const simd_view &Y) {                \
    return (element_type)X RELOP(element_type) Y;                              \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP
};

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
