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

  using element_type = typename ShapeTy::element_type;

  /// The simd type if reading this simd_view object.
  using value_type = simd<element_type, length>;

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
      const simd_view &X, const element_type &Y) {                             \
    return X RELOP(value_type) Y;                                              \
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
/// \ingroup sycl_esimd
template <typename BaseTy, typename T, int StrideY, int StrideX>
class simd_view<BaseTy, region1d_scalar_t<T, StrideY, StrideX>>
    : public detail::simd_view_impl<
          BaseTy, region1d_scalar_t<T, StrideY, StrideX>,
          simd_view<BaseTy, region1d_scalar_t<T, StrideY, StrideX>>> {
  template <typename, int> friend class simd;
  template <typename, typename, typename> friend class detail::simd_view_impl;

public:
  using RegionTy = region1d_scalar_t<T, StrideY, StrideX>;
  using BaseClass =
      detail::simd_view_impl<BaseTy, RegionTy, simd_view<BaseTy, RegionTy>>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;
  static_assert(1 == length, "length of this view is not equal to 1");
  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = T;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  operator element_type() const {
    const auto v = BaseClass::read();
    return v[0];
  }

  using BaseClass::operator=;

#define DEF_RELOP(RELOP)                                                       \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X,                  \
                                          const simd_view &Y) {                \
    return (element_type)X RELOP(element_type) Y;                              \
  }                                                                            \
  template <typename T1, typename = sycl::detail::enable_if_t<                 \
                             detail::is_esimd_scalar<T1>::value &&             \
                             detail::is_vectorizable_v<T1>::value>>            \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X, T1 Y) {          \
    return (element_type)X RELOP Y;                                            \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP
};

// TODO: remove code duplication in two class specializations for a simd_view
// with a single element

/// This is a specialization of nested simd_view class with a single element.
/// Objects of such a class are created in the following situation:
///   simd<int, 4> v = 1;
///   auto v1 = v.select<2, 1>(0);
///   auto v2 = v1[0]; // simd_view of a nested region for a single element
template <typename BaseTy, typename T, int StrideY, int StrideX,
          typename NestedRegion>
class simd_view<BaseTy,
                std::pair<region1d_scalar_t<T, StrideY, StrideX>, NestedRegion>>
    : public detail::simd_view_impl<
          BaseTy,
          std::pair<region1d_scalar_t<T, StrideY, StrideX>, NestedRegion>,
          simd_view<BaseTy, std::pair<region1d_scalar_t<T, StrideY, StrideX>,
                                      NestedRegion>>> {
  template <typename, int> friend class simd;
  template <typename, typename, typename> friend class detail::simd_view_impl;

public:
  using RegionTy =
      std::pair<region1d_scalar_t<T, StrideY, StrideX>, NestedRegion>;
  using BaseClass =
      detail::simd_view_impl<BaseTy, RegionTy, simd_view<BaseTy, RegionTy>>;
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;
  static_assert(1 == length, "length of this view is not equal to 1");
  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = T;

private:
  simd_view(BaseTy &Base, RegionTy Region) : BaseClass(Base, Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : BaseClass(Base, Region) {}

public:
  operator element_type() const {
    const auto v = BaseClass::read();
    return v[0];
  }

  using BaseClass::operator=;

#define DEF_RELOP(RELOP)                                                       \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X,                  \
                                          const simd_view &Y) {                \
    return (element_type)X RELOP(element_type) Y;                              \
  }                                                                            \
  template <typename T1, typename = sycl::detail::enable_if_t<                 \
                             detail::is_esimd_scalar<T1>::value &&             \
                             detail::is_vectorizable_v<T1>::value>>            \
  ESIMD_INLINE friend bool operator RELOP(const simd_view &X, T1 Y) {          \
    return (element_type)X RELOP Y;                                            \
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
