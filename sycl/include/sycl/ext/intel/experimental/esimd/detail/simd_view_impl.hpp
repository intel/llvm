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
#include <sycl/ext/intel/experimental/esimd/detail/test_proxy.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/type_format.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

/// @addtogroup sycl_esimd_core
/// @{

/// Base class for "simd view" types.
/// It is an internal class implementing basic functionality inherited by all
/// simd_view specializations. Objects of this type and its subclasses provide a
/// a "view" into objects of types inheriting from simd_obj_impl - @ref simd or
/// @ref simd_mask. Such "viewed into" object are called view "targets". The
/// element type of the view and the target object may differ. The view can also
/// span a subset of target's elements (region) - e.g. in a strided manner.
/// These view parameters - element type and region - are defined by the
/// region type the view class is templated on. simd_view_impl objects are never
/// created directly, only via subclassing.
/// @tparam BaseTy The type of the target object. Auto-deduced.
/// @tparam RegionTy The type defining compile-time known characteristics of the
///   viewed region within the target object. Auto-deduced.
///
template <typename BaseTy,
          typename RegionTy =
              region1d_t<typename BaseTy::element_type, BaseTy::length, 1>>
class simd_view_impl {
public:
  /// The only type which is supposed to extend this one and be used in user
  /// code.
  using Derived = simd_view<BaseTy, RegionTy>;

protected:
  /// @cond EXCLUDE

  template <typename, int, class, class> friend class simd_obj_impl;
  template <typename, int> friend class simd;
  template <typename, typename> friend class simd_view_impl;
  template <typename, int> friend class simd_mask_impl;

  static_assert(is_simd_obj_impl_derivative_v<BaseTy>);

protected:
  // Deduce the corresponding value type from its region type.
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  using base_type = BaseTy;
  template <typename ElT, int N>
  using get_simd_t = construct_a_simd_type_t<base_type, ElT, N>;

  /// The region type of this class.
  using region_type = RegionTy;

  /// @endcond EXCLUDE

public:
  /// Element type of this view, may differ from the element type of the target
  /// object.
  using element_type = typename ShapeTy::element_type;

  /// Corresponding "raw" (storage) type for the element type.
  using raw_element_type = __raw_t<element_type>;

  /// The simd type of the viewed region of the target object.
  using value_type = get_simd_t<element_type, length>;

  /// The underlying builtin vector type of the the viewed region.
  using raw_vector_type = vector_type_t<__raw_t<element_type>, length>;

private:
  /// @cond EXCLUDE

  Derived &cast_this_to_derived() { return reinterpret_cast<Derived &>(*this); }

protected:
  simd_view_impl(BaseTy &Base, RegionTy Region)
      : M_base(Base), M_region(Region) {}

  simd_view_impl(BaseTy &Base) : M_base(Base), M_region(RegionTy(0)) {}

  /// @endcond EXCLUDE

public:
  /// Default copy constructor.
  simd_view_impl(const simd_view_impl &Other) = default;

  /// Default move constructor.
  simd_view_impl(simd_view_impl &&Other) = default;

  /// Implicit conversion to a simd object with potentially different element
  /// type. Reads the viewed region from the target, converts elements to the
  /// requested type and returns as a simd object. Available only then the type
  /// of the view target is simd.
  /// @tparam ToTy The element type of the result.
  template <typename ToTy, class T = BaseTy,
            class = std::enable_if_t<is_simd_type_v<T>>>
  inline operator simd<ToTy, length>() const {
    if constexpr (std::is_same_v<element_type, ToTy>)
      return read();
    else
      return convert_vector<ToTy, element_type, length>(read().data());
  }

  /// Implicit conversion to a simd_mask object. Reads the viewed region from
  /// the target and returns it as a simd object. Available only then the type
  /// of the view target is simd_mask.
  template <class T = BaseTy, class = std::enable_if_t<is_simd_mask_type_v<T>>>
  inline operator simd_mask_type<length>() const {
    return read();
  }

  /// Tells whether this view is 1-dimensional.
  static constexpr bool is1D() { return !ShapeTy::Is_2D; }
  /// Tells whether this view is 2-dimensional.
  static constexpr bool is2D() { return ShapeTy::Is_2D; }
  /// Get number of elements in the view along X dimension.
  static constexpr int getSizeX() { return ShapeTy::Size_x; }
  /// Get element stride of the view along X dimension.
  static constexpr int getStrideX() { return ShapeTy::Stride_x; }
  /// Get number of elements in the view along Y dimension.
  static constexpr int getSizeY() { return ShapeTy::Size_y; }
  /// Get element stride of the view along Y dimension.
  static constexpr int getStrideY() { return ShapeTy::Stride_y; }

  /// Get the offset of the first element of the view within the parent object
  /// along X dimension.
  constexpr uint16_t getOffsetX() const {
    return getTopRegion(M_region).M_offset_x;
  }

  /// Get the offset of the first element of the view within the parent object
  /// along Y dimension.
  constexpr uint16_t getOffsetY() const {
    return getTopRegion(M_region).M_offset_y;
  }

  /// Reads the viewed region from the target w/o any conversion and returns as
  /// an object of the \c value_type type.
  /// @return The viewed target region as a @ref simd or @ref simd_mask object.
  value_type read() const {
    using BT = typename BaseTy::element_type;
    constexpr int BN = BaseTy::length;
    return value_type{readRegion<BT, BN>(M_base.data(), M_region)};
  }

  /// @return Viewed region's raw elements as a vector.
  typename value_type::raw_vector_type data() const { return read().data(); }

  /// Assigns a new value to the viewed target's region.
  /// @param Val The new value to assign.
  /// @return Reference to this object properly casted to the actual type
  /// subclassing the simd_view_impl.
  Derived &write(const value_type &Val) {
    M_base.writeRegion(M_region, Val.data());
    return cast_this_to_derived();
  }

  /// "Merges" the viewed target's region with given value according to a
  /// per-element mask.
  /// @param Val The new value to merge with.
  /// @param Mask The mask. Only elements in lanes with non-zero mask
  ///   predicate are assigned from corresponding \c Val elements.
  void merge(const value_type &Val, const simd_mask_type<length> &Mask) {
    merge(Val, read(), Mask);
  }

  /// "Merges" given values according to a per-element mask and writes the
  /// result to the viewed target's region.
  /// @param Val1 Value1 being merged.
  /// @param Val2 Value2 being merged.
  /// @param Mask The mask. non-zero in a mask's lane tells to take
  ///   corresponding element from \c Val1, zero - from \c Val2.
  void merge(const value_type &Val1, value_type Val2,
             const simd_mask_type<length> &Mask) {
    Val2.merge(Val1, Mask);
    write(Val2.read());
  }

  /// Create a 1-dimensional view of the target region.
  /// @tparam EltTy The element type of the new view.
  /// @return a new simd_view object which spans the same target region, but
  ///   with the new element type and potentially different number of elements,
  ///   if the sizes of this view's element type and the new one don't match.
  template <typename EltTy> auto bit_cast_view() {
    using TopRegionTy = detail::compute_format_type_t<Derived, EltTy>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  /// Create a 2-dimensional view of the target region.
  /// <code>sizeof(EltTy)*Height*Width</code> must be equal to the byte size of
  /// the target region.
  /// @tparam ElTy The element type of the new view.
  /// @tparam Height Height of the new view in rows.
  /// @tparam Width Width of the new view in elements.
  /// @return A new 2D \c simd_view object which spans the same target region,
  ///   but potentially with a different element type and different number of
  ///   elements, if the sizes of this object's element type and the new one
  ///   don't match.
  template <typename EltTy, int Height, int Width> auto bit_cast_view() {
    using TopRegionTy =
        detail::compute_format_type_2d_t<Derived, EltTy, Height, Width>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0, 0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  /// 1D region select. Selects a 1D subregion in the target region.
  ///
  /// @tparam Size The number of elements to be selected.
  /// @tparam Stride Distance in elements between two consecutive elements.
  /// @param Offset is the starting element offset.
  /// @return \c simd_view representing the subregion.
  template <int Size, int Stride, typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  auto select(uint16_t Offset = 0) {
    using TopRegionTy = region1d_t<element_type, Size, Stride>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(Offset);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  // clang-format off
  /// 2D region select. Selects a 2D subregion in the target region.
  /// The target region must be a 2D region, for example, a result of calling
  /// the 2D version of \c simd_obj_impl::bit_cast_view. Code snippet below
  /// shows example of naive implementation of matrix multiplication of two
  /// matrices:
  /// - \c tile_a, [\c Wm x \c Wk] elements selected from object \c a, with
  ///   offsets \c off_m, \c off_k and strides \c Sm and \c Sk along the Y and X
  ///   dimensions
  /// - \c tile_b, [\c Wk x \c Wn] elements selected from object \c b, with
  ///   offsets \c off_k, \c off_n and strides \c Sk and \c Sn along the Y and X
  ///   dimensions
  /// @code{.cpp}
  /// simd<T, M * K> a(mat_a);
  /// simd<T, K * N> b(mat_b);
  /// simd<T, M * N> c(mat_c);
  ///
  /// auto tile_a = a.template bit_cast_view<T, M, K>().template select<Wm, Sm, Wk, Sk>(off_m, off_k);
  /// auto tile_b = b.template bit_cast_view<T, K, N>().template select<Wk, Sk, Wn, Sn>(off_k, off_n);
  /// auto tile_c = c.template bit_cast_view<T, M, N>().template select<Wm, Sm, Wn, Sn>(off_m, off_n);
  ///
  /// for (int m = 0; m < Wm; m++) {
  ///   for (int n = 0; n < Wn; n++) {
  ///     tile_c.template select<1, 1, 1, 1>(m, n) +=
  ///       reduce<T>(tile_a.row(m) * tile_b.column(n), std::plus<>{});
  ///   }
  /// }
  /// c.copy_to(mat_c);
  /// @endcode
  ///
  /// @tparam SizeY The number of elements to be selected in Y-dimension.
  /// @tparam StrideY Distance in elements between two consecutive elements
  ///  in Y-dimension.
  /// @tparam SizeX The number of elements to be selected in X-dimension.
  /// @tparam StrideX Distance in elements between two consecutive elements
  ///   in X-dimension.
  /// @param OffsetX Starting element offset in X-dimension.
  /// @param OffsetY Starting element offset in Y-dimension.
  /// @return 2D simd_view of the subregion.
  // clang-format on
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
#define __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(BINOP, OPASSIGN, COND)             \
                                                                               \
  /* OPASSIGN simd_obj_impl */                                                 \
  template <class T1, int N1, class SimdT1, class T = element_type,            \
            class SimdT = BaseTy,                                              \
            class =                                                            \
                std::enable_if_t<(is_simd_type_v<SimdT> ==                     \
                                  is_simd_type_v<SimdT1>)&&(N1 == length) &&   \
                                 COND>>                                        \
  Derived &operator OPASSIGN(const simd_obj_impl<T1, N1, SimdT1> &RHS) {       \
    auto Res = read() BINOP RHS;                                               \
    write(Res);                                                                \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /* OPASSIGN simd_view_impl */                                                \
  template <class SimdT1, class RegionT1,                                      \
            class T1 = typename __SEIEE::shape_type<RegionT1>::element_type,   \
            class T = element_type, class SimdT = BaseTy,                      \
            class = std::enable_if_t<                                          \
                (is_simd_type_v<SimdT> == is_simd_type_v<SimdT1>)&&(           \
                    length == __SEIEE::shape_type<RegionT1>::length) &&        \
                COND>>                                                         \
  Derived &operator OPASSIGN(const simd_view_impl<SimdT1, RegionT1> &RHS) {    \
    *this OPASSIGN RHS.read();                                                 \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /* OPASSIGN scalar */                                                        \
  template <class T1, class T = element_type, class SimdT = BaseTy,            \
            class = std::enable_if_t<COND>>                                    \
  Derived &operator OPASSIGN(T1 RHS) {                                         \
    auto Res = read() BINOP RHS;                                               \
    write(Res);                                                                \
    return cast_this_to_derived();                                             \
  }

#define __ESIMD_BITWISE_OP_FILTER std::is_integral_v<T> &&std::is_integral_v<T1>
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(^, ^=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(|, |=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(&, &=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(%, %=, __ESIMD_BITWISE_OP_FILTER)
#undef __ESIMD_BITWISE_OP_FILTER

#define __ESIMD_SHIFT_OP_FILTER                                                \
  std::is_integral_v<T> &&std::is_integral_v<T1> &&is_simd_type_v<SimdT>

  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(<<, <<=, __ESIMD_SHIFT_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(>>, >>=, __ESIMD_SHIFT_OP_FILTER)
#undef __ESIMD_SHIFT_OP_FILTER

#define __ESIMD_ARITH_OP_FILTER                                                \
  is_valid_simd_elem_type_v<T> &&is_valid_simd_elem_type_v<T1>                 \
      &&is_simd_type_v<SimdT>

  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(+, +=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(-, -=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(*, *=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN(/, /=, __ESIMD_ARITH_OP_FILTER)

#undef __ESIMD_ARITH_OP_FILTER
#undef __ESIMD_DEF_SIMD_VIEW_IMPL_OPASSIGN

#define __ESIMD_DEF_UNARY_OP(UNARY_OP, COND)                                   \
  template <class T = element_type, class SimdT = BaseTy,                      \
            class = std::enable_if_t<COND>>                                    \
  auto operator UNARY_OP() {                                                   \
    auto V = UNARY_OP(read().data());                                          \
    return get_simd_t<element_type, length>(V);                                \
  }
  __ESIMD_DEF_UNARY_OP(~, std::is_integral_v<T> &&is_simd_type_v<SimdT>)
  __ESIMD_DEF_UNARY_OP(+, is_simd_type_v<SimdT>)
  __ESIMD_DEF_UNARY_OP(-, is_simd_type_v<SimdT>)

#undef __ESIMD_DEF_UNARY_OP

  /// Unary logical negeation operator. Applies only to integer element types.
  template <class T = element_type,
            class = std::enable_if_t<std::is_integral_v<T>>>
  auto operator!() {
    using MaskVecT = typename simd_mask_type<length>::raw_vector_type;
    auto V = read().data() == 0;
    return simd_mask_type<length>{__builtin_convertvector(V, MaskVecT) &
                                  MaskVecT(1)};
  }

  /// Copy assignment. Updates the target region viewed by this object.
  /// @param Other The source object.
  simd_view_impl &operator=(const simd_view_impl &Other) {
    return write(Other.read());
  }

  /// Assignment from an object of the derived class. Updates the target
  /// region viewed by this object.
  /// @param Other The source object.
  /// @return This object cast to the derived class.
  Derived &operator=(const Derived &Other) { return write(Other.read()); }

  /// Assignment from a \c value_type object - \c simd or \c simd_mask.
  /// Updates the target region viewed by this object.
  /// @param Val The source object.
  /// @return This object cast to the derived class.
  Derived &operator=(const value_type &Val) { return write(Val); }

  /// Assignment from an rvalue object of the derived class. Updates the target
  /// region viewed by this object.
  /// @param Other The source rvalue object.
  /// @return This object cast to the derived class.
  Derived &operator=(Derived &&Other) {
    __esimd_move_test_proxy(Other);
    return write(Other.read());
  }

  /// Move assignment operator. Updates the target region viewed by this object.
  simd_view_impl &operator=(simd_view_impl &&Other) {
    __esimd_move_test_proxy(Other);
    return write(Other.read());
  }

  /// Assignment with element type conversion from a vector object. Updates the
  /// target region viewed by this object. Available only when the length
  /// (number of elements) of the source object matches the length of the target
  /// region.
  /// @tparam T The source vector element type. Auto-deduced.
  /// @tparam N The source vector length. Auto-deduced.
  /// @param Other The source vector.
  /// @return This object cast to the derived class.
  template <class T, int N, class SimdT,
            class = std::enable_if_t<(is_simd_type_v<SimdT> ==
                                      is_simd_type_v<BaseTy>)&&(length ==
                                                                SimdT::length)>>
  Derived &operator=(const simd_obj_impl<T, N, SimdT> &Other) {
    return write(convert_vector<element_type, typename SimdT::element_type, N>(
        Other.data()));
  }

  /// Broadcast assignment of a scalar with conversion. Updates the target
  /// region viewed by this object.
  /// @tparam T1 The type of the scalar.
  /// @param RHS The scalar.
  /// @return This object cast to the derived class.
  template <class T1, class = std::enable_if_t<is_valid_simd_elem_type_v<T1>>>
  Derived &operator=(T1 RHS) {
    return write(value_type(convert_scalar<element_type>(RHS)));
  }

  /// Prefix increment. Updates the target region viewed by this object.
  /// @return This object cast to the derived class.
  Derived &operator++() {
    *this += 1;
    return cast_this_to_derived();
  }

  /// Postfix increment.
  /// @return New vector object, whose element values are incremented values of
  /// the target region elements.
  value_type operator++(int) {
    value_type Ret(read());
    operator++();
    return Ret;
  }

  /// Prefix decrement. Updates the target region viewed by this object.
  /// @return This object cast to the derived class.
  Derived &operator--() {
    *this -= 1;
    return cast_this_to_derived();
  }

  /// Postfix decrement.
  /// @return New vector object, whose element values are decremented values of
  /// the target region elements.
  value_type operator--(int) {
    value_type Ret(read());
    operator--();
    return Ret;
  }

  /// Reference a row from a 2D region. Available only if this object is 2D.
  /// @param i Row index.
  /// @return A 2D view of a region representing i'th row of the target region.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto row(int i) {
    return select<1, 1, getSizeX(), 1>(i, 0)
        .template bit_cast_view<element_type>();
  }

  /// Reference a column from a 2D region. Available only if this object is 2D.
  /// @param i Column index.
  /// @return A 2D view of a region representing i'th column of the target
  ///   region.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto column(int i) {
    return select<getSizeY(), 1, 1, 1>(0, i);
  }

  /// Read a single element from the target 1D region.
  /// @param i Element index.
  /// @return Element value.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  element_type operator[](int i) const {
    const auto v = read();
    return v[i];
  }

  /// Return a writeable view of a single element in the target 1D region.
  /// @param i Element index.
  /// @return A new 1D view of the element. Can be used to update it.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  auto operator[](int i) {
    return select<1, 1>(i);
  }

  /// Applies simd_obj_impl::replicate to the target region.
  template <int Rep> get_simd_t<element_type, Rep> replicate() {
    return read().template replicate<Rep>();
  }

  /// Shortcut to replicate_vs_w<int Rep, int Vs, int W>(uint16_t) with
  /// with \c Vs = 0. Used to replicate the same elements chunk multiple times.
  /// Used for 1D views.
  template <int Rep, int W>
  get_simd_t<element_type, Rep * W> replicate_w(uint16_t OffsetX) {
    return replicate_vs_w<Rep, 0, W>(0, OffsetX);
  }

  /// Shortcut to replicate_vs_w<int Rep, int Vs, int W>(uint16_t, uint16_t)
  /// with \c Vs = 0. Used to replicate the same elements chunk multiple times.
  /// Used for 2D views.
  template <int Rep, int W>
  get_simd_t<element_type, Rep * W> replicate_w(uint16_t OffsetY,
                                                uint16_t OffsetX) {
    return replicate_vs_w<Rep, 0, W>(OffsetY, OffsetX);
  }

  // clang-format off
  /// Shortcut to
  /// replicate_vs_w_hs<int Rep, int VS, int W, int Hs>(uint16_t OffsetY, uint16_t OffsetX)
  /// with \c Hs = 1 and \c OffsetY = 0.
  // clang-format on
  template <int Rep, int VS, int W>
  get_simd_t<element_type, Rep * W> replicate_vs_w(uint16_t OffsetX) {
    return replicate_vs_w_hs<Rep, VS, W, 1>(0, OffsetX);
  }

  // clang-format off
  /// Shortcut to
  /// replicate_vs_w_hs<int Rep, int VS, int W, int Hs>(uint16_t OffsetY, uint16_t OffsetX)
  /// with \c Hs = 1
  // clang-format on
  template <int Rep, int VS, int W>
  get_simd_t<element_type, Rep * W> replicate_vs_w(uint16_t OffsetY,
                                                   uint16_t OffsetX) {
    return replicate_vs_w_hs<Rep, VS, W, 1>(OffsetY, OffsetX);
  }

  /// Applies simd_obj_impl::replicate_vs_w_hs<int Rep, int VS, int W, int HS>
  /// to the target region.
  template <int Rep, int VS, int W, int HS>
  get_simd_t<element_type, Rep * W> replicate_vs_w_hs(uint16_t OffsetX) {
    return read().template replicate_vs_w_hs<Rep, VS, W, HS>(OffsetX);
  }

  /// Applies simd_obj_impl::replicate_vs_w_hs<int Rep, int VS, int W, int HS>
  /// to the target region. The offset is calculated as
  /// <code>OffsetY * RowSize + OffsetX</code> where \c RowSize is the size of
  /// the X dimension if this object is 2D or \c 0 otherwise.
  template <int Rep, int VS, int W, int HS>
  get_simd_t<element_type, Rep * W> replicate_vs_w_hs(uint16_t OffsetY,
                                                      uint16_t OffsetX) {
    constexpr int RowSize = is2D() ? getSizeX() : 0;
    return read().template replicate_vs_w_hs<Rep, VS, W, HS>(OffsetY * RowSize +
                                                             OffsetX);
  }

  /// Applies simd_obj_impl::any operation to the target region.
  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return read().any();
  }

  /// Applies simd_obj_impl::all operation to the target region.
  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return read().all();
  }

  /// @cond EXCLUDE
public:
  // Getter for the test proxy member, if enabled
  __ESIMD_DECLARE_TEST_PROXY_ACCESS

protected:
  // The reference to the base object, which must be a simd object
  BaseTy &M_base;

  // The test proxy if enabled
  __ESIMD_DECLARE_TEST_PROXY

  // The region applied on the base object. Its type could be
  // - region1d_t
  // - region2d_t
  // - std::pair<top_region_type, base_region_type>
  //
  RegionTy M_region;
  /// @endcond EXCLUDE
};

/// @} sycl_esimd_core

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
