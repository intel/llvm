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

/// The simd_view base class.
/// It is an internal class implementing basic functionality of simd_view.
///
template <typename BaseTy,
          typename RegionTy =
              region1d_t<typename BaseTy::element_type, BaseTy::length, 1>>
class simd_view_impl {
  using Derived = simd_view<BaseTy, RegionTy>;
  template <typename, int, class, class> friend class simd_obj_impl;
  template <typename, int> friend class simd;
  template <typename, typename> friend class simd_view_impl;
  template <typename, int> friend class simd_mask_impl;

public:
  static_assert(is_simd_obj_impl_derivative_v<BaseTy>);
  // Deduce the corresponding value type from its region type.
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  using base_type = BaseTy;
  template <typename ElT, int N>
  using get_simd_t = construct_a_simd_type_t<base_type, ElT, N>;

  /// The region type of this class.
  using region_type = RegionTy;

  /// The element type of this class, which could be different from the element
  /// type of the base object type.
  using element_type = typename ShapeTy::element_type;
  using raw_element_type = __raw_t<element_type>;

  /// The simd type if reading the object.
  using value_type = get_simd_t<element_type, length>;

  /// The underlying builtin vector type backing the value read from the object.
  using raw_vector_type = vector_type_t<__raw_t<element_type>, length>;

private:
  Derived &cast_this_to_derived() { return reinterpret_cast<Derived &>(*this); }

protected:
  simd_view_impl(BaseTy &Base, RegionTy Region)
      : M_base(Base), M_region(Region) {}

  simd_view_impl(BaseTy &Base) : M_base(Base), M_region(RegionTy(0)) {}

public:
  /// Default copy constructor.
  simd_view_impl(const simd_view_impl &Other) = default;

  /// Default move constructor.
  simd_view_impl(simd_view_impl &&Other) = default;

  /// Implicit conversion to simd type.
  template <typename ToTy, class T = BaseTy,
            class = std::enable_if_t<is_simd_type_v<T>>>
  inline operator simd<ToTy, length>() const {
    if constexpr (std::is_same_v<element_type, ToTy>)
      return read();
    else
      return convert_vector<ToTy, element_type, length>(read().data());
  }

  /// Implicit conversion to simd_mask_impl type, if element type is compatible.
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

  /// Read the object.
  value_type read() const {
    using BT = typename BaseTy::element_type;
    constexpr int BN = BaseTy::length;
    return value_type{readRegion<BT, BN>(M_base.data(), M_region)};
  }

  typename value_type::raw_vector_type data() const { return read().data(); }

  /// Write to this object.
  Derived &write(const value_type &Val) {
    M_base.writeRegion(M_region, Val.data());
    return cast_this_to_derived();
  }

  /// Whole region update with predicates.
  void merge(const value_type &Val, const simd_mask_type<length> &Mask) {
    merge(Val, read(), Mask);
  }

  void merge(const value_type &Val1, value_type Val2,
             const simd_mask_type<length> &Mask) {
    Val2.merge(Val1, Mask);
    write(Val2.read());
  }

  /// View this object in a different element type.
  template <typename EltTy> auto bit_cast_view() {
    using TopRegionTy = detail::compute_format_type_t<Derived, EltTy>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
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

  /// Assignment operators.
  simd_view_impl &operator=(const simd_view_impl &Other) {
    return write(Other.read());
  }

  Derived &operator=(const Derived &Other) { return write(Other.read()); }

  Derived &operator=(const value_type &Val) { return write(Val); }

  /// Move assignment operator.
  Derived &operator=(Derived &&Other) {
    __esimd_move_test_proxy(Other);
    return write(Other.read());
  }
  simd_view_impl &operator=(simd_view_impl &&Other) {
    __esimd_move_test_proxy(Other);
    return write(Other.read());
  }

  template <class T, int N, class SimdT,
            class = std::enable_if_t<(is_simd_type_v<SimdT> ==
                                      is_simd_type_v<BaseTy>)&&(length ==
                                                                SimdT::length)>>
  Derived &operator=(const simd_obj_impl<T, N, SimdT> &Other) {
    return write(convert_vector<element_type, typename SimdT::element_type, N>(
        Other.data()));
  }

  template <class T1, class = std::enable_if_t<is_valid_simd_elem_type_v<T1>>>
  Derived &operator=(T1 RHS) {
    return write(value_type(convert_scalar<element_type>(RHS)));
  }

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
    return select<1, 1, getSizeX(), 1>(i, 0)
        .template bit_cast_view<element_type>();
  }

  /// Reference a column from a 2D region.
  /// \return a 2D region.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is2D()>>
  auto column(int i) {
    return select<getSizeY(), 1, 1, 1>(0, i);
  }

  /// Read a single element from a 1D region, by value only.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  element_type operator[](int i) const {
    const auto v = read();
    return v[i];
  }

  /// Return a writeable view of a single element.
  template <typename T = Derived,
            typename = sycl::detail::enable_if_t<T::is1D()>>
  auto operator[](int i) {
    return select<1, 1>(i);
  }

  /// Replicate. Create a new simd object from a subset of elements
  /// referred to by this \c simd_view_impl object.
  /// \tparam Rep is number of times region has to be replicated.
  template <int Rep> get_simd_t<element_type, Rep> replicate() {
    return read().template replicate<Rep>();
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W>
  get_simd_t<element_type, Rep * W> replicate_w(uint16_t OffsetX) {
    return replicate_vs_w<Rep, 0, W>(0, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W>
  get_simd_t<element_type, Rep * W> replicate_w(uint16_t OffsetY,
                                                uint16_t OffsetX) {
    return replicate_vs_w<Rep, 0, W>(OffsetY, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  get_simd_t<element_type, Rep * W> replicate_vs_w(uint16_t OffsetX) {
    return replicate_vs_w_hs<Rep, VS, W, 1>(0, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  get_simd_t<element_type, Rep * W> replicate_vs_w(uint16_t OffsetY,
                                                   uint16_t OffsetX) {
    return replicate_vs_w_hs<Rep, VS, W, 1>(OffsetY, OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS is horizontal stride of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  get_simd_t<element_type, Rep * W> replicate_vs_w_hs(uint16_t OffsetX) {
    return read().template replicate_vs_w_hs<Rep, VS, W, HS>(OffsetX);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS is vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS is horizontal stride of src region to replicate.
  /// \param OffsetX is column offset in number of elements in src region.
  /// \param OffsetY is row offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  get_simd_t<element_type, Rep * W> replicate_vs_w_hs(uint16_t OffsetY,
                                                      uint16_t OffsetX) {
    constexpr int RowSize = is2D() ? getSizeX() : 0;
    return read().template replicate_vs_w_hs<Rep, VS, W, HS>(OffsetY * RowSize +
                                                             OffsetX);
  }

  /// 'any' operation.
  ///
  /// \return 1 if any element is set, 0 otherwise.
  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return read().any();
  }

  /// 'all' operation.
  ///
  /// \return 1 if all elements are set, 0 otherwise.
  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return read().all();
  }

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
};

/// @} sycl_esimd_core

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
