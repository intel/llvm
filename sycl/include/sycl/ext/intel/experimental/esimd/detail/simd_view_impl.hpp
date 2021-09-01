//==------------ - simd_view_impl.hpp - DPC++ Explicit SIMD API   ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation detail of Explicit SIMD vector view class.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

/// The simd_view base class.
/// It is an internal class implementing basic functionality of simd_view.
///
/// \ingroup sycl_esimd
template <typename BaseTy, typename RegionTy, typename Derived>
class simd_view_impl {
  template <typename, int> friend class simd;
  template <typename, typename, typename> friend class simd_view_impl;

public:
  static_assert(!detail::is_simd_view_v<BaseTy>::value);
  // Deduce the corresponding value type from its region type.
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  /// The simd type if reading the object.
  using value_type = simd<typename ShapeTy::element_type, length>;

  /// The underlying builtin value type
  using vector_type =
      detail::vector_type_t<typename ShapeTy::element_type, length>;

  /// The region type of this class.
  using region_type = RegionTy;

  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = typename ShapeTy::element_type;

  /// @{
  /// Constructors.

private:
  Derived &cast_this_to_derived() { return reinterpret_cast<Derived &>(*this); }

protected:
  simd_view_impl(BaseTy &Base, RegionTy Region)
      : M_base(Base), M_region(Region) {}
  simd_view_impl(BaseTy &&Base, RegionTy Region)
      : M_base(Base), M_region(Region) {}

public:
  // Default copy and move constructors.
  simd_view_impl(const simd_view_impl &Other) = default;
  simd_view_impl(simd_view_impl &&Other) = default;
  /// @}

  /// Conversion to simd type.
  template <typename ToTy> operator simd<ToTy, length>() const {
    if constexpr (std::is_same<element_type, ToTy>::value)
      return read();
    else
      return convert<ToTy, element_type, length>(read());
  }

  /// @{
  /// Assignment operators.
  simd_view_impl &operator=(const simd_view_impl &Other) {
    return write(Other.read());
  }
  simd_view_impl &operator=(const value_type &Val) { return write(Val); }
  /// @}

  /// Move assignment operator.
  simd_view_impl &operator=(simd_view_impl &&Other) {
    return write(Other.read());
  }

  /// @{
  /// Region accessors.
  static constexpr bool is1D() { return !ShapeTy::Is_2D; }
  static constexpr bool is2D() { return ShapeTy::Is_2D; }
  static constexpr int getSizeX() { return ShapeTy::Size_x; }
  static constexpr int getStrideX() { return ShapeTy::Stride_x; }
  static constexpr int getSizeY() { return ShapeTy::Size_y; }
  static constexpr int getStrideY() { return ShapeTy::Stride_y; }
  constexpr uint16_t getOffsetX() const {
    return getTopRegion(M_region).M_offset_x;
  }
  constexpr uint16_t getOffsetY() const {
    return getTopRegion(M_region).M_offset_y;
  }
  /// @}

  /// Read the object.
  value_type read() const {
    using BT = typename BaseTy::element_type;
    constexpr int BN = BaseTy::length;
    return detail::readRegion<BT, BN>(M_base.data(), M_region);
  }

  /// Write to this object.
  Derived &write(const value_type &Val) {
    M_base.writeRegion(M_region, Val.data());
    return cast_this_to_derived();
  }

  /// @{
  /// Whole region update with predicates.
  void merge(const value_type &Val, const mask_type_t<length> &Mask) {
    merge(Val, read(), Mask);
  }

  void merge(const value_type &Val1, value_type Val2,
             const mask_type_t<length> &Mask) {
    Val2.merge(Val1, Mask);
    write(Val2.read());
  }
  /// @}

  /// View this object in a different element type.
  template <typename EltTy> auto bit_cast_view() {
    using TopRegionTy = detail::compute_format_type_t<Derived, EltTy>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  template <typename EltTy>
  __SYCL_DEPRECATED("use simd_view::bit_cast_view.")
  auto format() {
    return bit_cast_view<EltTy>();
  }

  /// View as a 2-dimensional simd_view.
  template <typename EltTy, int Height, int Width> auto bit_cast_view() {
    using TopRegionTy =
        detail::compute_format_type_2d_t<Derived, EltTy, Height, Width>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0, 0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  template <typename EltTy, int Height, int Width>
  __SYCL_DEPRECATED("use simd_view::bit_cast_view.")
  auto format() {
    return bit_cast_view<EltTy, Height, Width>();
  }

  /// 1D region select, apply a region on top of this object.
  ///
  /// \param Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the representing region object.
  template <int Size, int Stride, typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  auto select(uint16_t Offset = 0) {
    using TopRegionTy = region1d_t<element_type, Size, Stride>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(Offset);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  /// 2D region select, apply a region on top of this object.
  ///
  /// \tparam SizeX is the number of elements to be selected in X-dimension.
  /// \tparam StrideX is the element distance between two consecutive elements
  /// in X-dimension.
  /// \tparam SizeY is the number of elements to be selected in Y-dimension.
  /// \tparam StrideY is the element distance between two consecutive elements
  /// Y-dimension.
  /// \param OffsetX is the starting element offset in X-dimension.
  /// \param OffsetY is the starting element offset in Y-dimension.
  /// \return the representing region object.
  template <int SizeY, int StrideY, int SizeX, int StrideX,
            typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto select(uint16_t OffsetY = 0, uint16_t OffsetX = 0) {
    using TopRegionTy =
        region2d_t<element_type, SizeY, StrideY, SizeX, StrideX>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(OffsetY, OffsetX);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

#define DEF_BINOP(BINOP, OPASSIGN)                                             \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BINOP(const Derived &X,                    \
                                          const value_type &Y) {               \
    using ComputeTy = detail::compute_type_t<value_type>;                      \
    auto V0 =                                                                  \
        detail::convert<typename ComputeTy::vector_type>(X.read().data());     \
    auto V1 = detail::convert<typename ComputeTy::vector_type>(Y.data());      \
    auto V2 = V0 BINOP V1;                                                     \
    return ComputeTy(V2);                                                      \
  }                                                                            \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BINOP(const Derived &X,                    \
                                          const element_type &Y) {             \
    return X BINOP(value_type) Y;                                              \
  }                                                                            \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BINOP(const value_type &X,                 \
                                          const Derived &Y) {                  \
    using ComputeTy = detail::compute_type_t<value_type>;                      \
    auto V0 = detail::convert<typename ComputeTy::vector_type>(X.data());      \
    auto V1 =                                                                  \
        detail::convert<typename ComputeTy::vector_type>(Y.read().data());     \
    auto V2 = V0 BINOP V1;                                                     \
    return ComputeTy(V2);                                                      \
  }                                                                            \
  ESIMD_INLINE friend auto operator BINOP(const Derived &X,                    \
                                          const Derived &Y) {                  \
    return (X BINOP Y.read());                                                 \
  }                                                                            \
  Derived &operator OPASSIGN(const value_type &RHS) {                          \
    using ComputeTy = detail::compute_type_t<value_type>;                      \
    auto V0 = detail::convert<typename ComputeTy::vector_type>(read().data()); \
    auto V1 = detail::convert<typename ComputeTy::vector_type>(RHS.data());    \
    auto V2 = V0 BINOP V1;                                                     \
    auto V3 = detail::convert<vector_type>(V2);                                \
    write(V3);                                                                 \
    return cast_this_to_derived();                                             \
  }                                                                            \
  Derived &operator OPASSIGN(const Derived &RHS) {                             \
    return (*this OPASSIGN RHS.read());                                        \
  }

  DEF_BINOP(+, +=)
  DEF_BINOP(-, -=)
  DEF_BINOP(*, *=)
  DEF_BINOP(/, /=)
  DEF_BINOP(%, %=)

#undef DEF_BINOP

#define DEF_BITWISE_OP(BITWISE_OP, OPASSIGN)                                   \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BITWISE_OP(const Derived &X,               \
                                               const value_type &Y) {          \
    static_assert(std::is_integral<element_type>(), "not integral type");      \
    auto V2 = X.read().data() BITWISE_OP Y.data();                             \
    return simd<element_type, length>(V2);                                     \
  }                                                                            \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BITWISE_OP(const Derived &X,               \
                                               const element_type &Y) {        \
    return X BITWISE_OP(value_type) Y;                                         \
  }                                                                            \
  template <class T1 = Derived, class = std::enable_if_t<T1::length != 1>>     \
  ESIMD_INLINE friend auto operator BITWISE_OP(const value_type &X,            \
                                               const Derived &Y) {             \
    static_assert(std::is_integral<element_type>(), "not integral type");      \
    auto V2 = X.data() BITWISE_OP Y.read().data();                             \
    return simd<element_type, length>(V2);                                     \
  }                                                                            \
  ESIMD_INLINE friend auto operator BITWISE_OP(const Derived &X,               \
                                               const Derived &Y) {             \
    return (X BITWISE_OP Y.read());                                            \
  }                                                                            \
  Derived &operator OPASSIGN(const value_type &RHS) {                          \
    static_assert(std::is_integral<element_type>(), "not integeral type");     \
    auto V2 = read().data() BITWISE_OP RHS.data();                             \
    auto V3 = detail::convert<vector_type>(V2);                                \
    write(V3);                                                                 \
    return cast_this_to_derived();                                             \
  }                                                                            \
  Derived &operator OPASSIGN(const Derived &RHS) {                             \
    return (*this OPASSIGN RHS.read());                                        \
  }
  DEF_BITWISE_OP(&, &=)
  DEF_BITWISE_OP(|, |=)
  DEF_BITWISE_OP(^, ^=)
  DEF_BITWISE_OP(>>, >>=)
  DEF_BITWISE_OP(<<, <<=)

#undef DEF_BITWISE_OP

#define DEF_UNARY_OP(UNARY_OP)                                                 \
  auto operator UNARY_OP() {                                                   \
    auto V = UNARY_OP(read().data());                                          \
    return simd<element_type, length>(V);                                      \
  }
  DEF_UNARY_OP(~)
  DEF_UNARY_OP(+)
  DEF_UNARY_OP(-)

#undef DEF_UNARY_OP

  // negation operator
  auto operator!() { return cast_this_to_derived() == 0; }

  // Operator ++, --
  Derived &operator++() {
    *this += 1;
    return cast_this_to_derived();
  }
  value_type operator++(int) {
    value_type Ret(read());
    operator++();
    return Ret;
  }
  Derived &operator--() {
    *this -= 1;
    return cast_this_to_derived();
  }
  value_type operator--(int) {
    value_type Ret(read());
    operator--();
    return Ret;
  }

  /// Reference a row from a 2D region.
  /// \return a 1D region.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto row(int i) {
    return select<1, 0, getSizeX(), 1>(i, 0)
        .template bit_cast_view<element_type>();
  }

  /// Reference a column from a 2D region.
  /// \return a 2D region.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto column(int i) {
    return select<getSizeY(), 1, 1, 0>(0, i);
  }

  /// Read a single element from a 1D region, by value only.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  element_type operator[](int i) const {
    const auto v = read();
    return v[i];
  }

  /// Read a single element from a 1D region, by value only.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  __SYCL_DEPRECATED("use operator[] form.")
  element_type operator()(int i) const {
    const auto v = read();
    return v[i];
  }

  /// Return a writeable view of a single element.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  auto operator[](int i) {
    return select<1, 0>(i);
  }

  /// Return a writeable view of a single element.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  __SYCL_DEPRECATED("use operator[] form.")
  auto operator()(int i) {
    return select<1, 0>(i);
  }

  /// \name Replicate
  /// Replicate simd instance given a simd_view
  /// @{
  ///

  /// \tparam Rep is number of times region has to be replicated.
  template <int Rep> simd<element_type, Rep> replicate() {
    return read().replicate<Rep>(0);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return replicate<Rep, 0, W>(0, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    return replicate<Rep, 0, W>(OffsetY, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return replicate<Rep, VS, W, 1>(0, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    return replicate<Rep, VS, W, 1>(OffsetY, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS is horizontal stride of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return read().template replicate<Rep, VS, W, HS>(OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS is horizontal stride of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    constexpr int RowSize = is2D() ? getSizeX() : 0;
    return read().template replicate<Rep, VS, W, HS>(OffsetY * RowSize +
                                                     OffsetX);
  }
  /// @}

  /// Any operation.
  ///
  /// \return 1 if any element is set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = BaseTy,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return read().any();
  }

  /// All operation.
  ///
  /// \return 1 if all elements are set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = BaseTy,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return read().all();
  }

protected:
  // The reference to the base object, which must be a simd object
  BaseTy &M_base;

  // The region applied on the base object. Its type could be
  // - region1d_t
  // - region2d_t
  // - std::pair<top_region_type, base_region_type>
  //
  RegionTy M_region;
};

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
