//==------------ - esimd_view.hpp - DPC++ Explicit SIMD API   --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD vector view APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/intel/esimd/detail/esimd_types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
namespace gpu {

//
// The reference class.
//
// This class represents a region applied to a base object, which
// must be a simd object.
//
template <typename BaseTy, typename RegionTy> class simd_view {
public:
  static_assert(!is_simd_view_v<BaseTy>::value);
  // Deduce the corresponding value type from its region type.
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int length = ShapeTy::Size_x * ShapeTy::Size_y;

  // The simd type if reading this simd_view object.
  using value_type = simd<typename ShapeTy::element_type, length>;

  // The underlying builtin value type
  using vector_type = vector_type_t<typename ShapeTy::element_type, length>;

  // The region type of this class.
  using region_type = RegionTy;

  // The element type of this class, which could be different from the element
  // type of the base object type.
  using element_type = typename ShapeTy::element_type;

  // TODO @rolandschulz
  // {quote}
  // Why is this and the next constructor public ? Those should only be called
  // internally by e.g.select, correct ?
  // {/quote}
  //
  // Constructors.
  simd_view(BaseTy &Base, RegionTy Region) : M_base(Base), M_region(Region) {}
  simd_view(BaseTy &&Base, RegionTy Region) : M_base(Base), M_region(Region) {}

  // TODO @rolandschulz
  // {quote}
  // Is this intentional not a correct copy constructor (would need to be const
  // for that)? I believe we agreed that simd_view would have a deleted copy and
  // move constructor.Why are they suddenly back ?
  // {/quote}
  // TODO @kbobrovs
  // copy constructor is still incorrect (no 'const'), move constructor is still
  // present.
  //
  // Disallow copy constructor for simd_view.
  simd_view(simd_view &Other) = delete;
  simd_view(simd_view &&Other)
      : M_base(Other.M_base), M_region(Other.M_region) {}

  // Conversion to simd value type.
  operator value_type() const { return read(); }

  // Assignment operators.
  simd_view &operator=(const simd_view &Other) { return write(Other.read()); }
  simd_view &operator=(const value_type &Val) { return write(Val); }

  // Region accessors.
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

  template <int Dim = 0> static constexpr int getSize() {
    static_assert(Dim <= is2D(), "region is not two-dimensional");
    return (Dim == 0) ? getSizeX() : getSizeY();
  }

  template <int Dim = 0> static constexpr int getStride() {
    static_assert(Dim <= is2D(), "region is not two-dimensional");
    return (Dim == 0) ? getStrideX() : getStrideY();
  }

  template <int Dim = 0> constexpr uint16_t getOffset() const {
    static_assert(Dim <= is2D(), "region is not two-dimensional");
    return (Dim == 0) ? getOffsetX() : getOffsetX();
  }

  // Read this simd_view object.
  value_type read() const {
    using BT = typename BaseTy::element_type;
    constexpr int BN = BaseTy::length;
    return readRegion<BT, BN>(M_base.data(), M_region);
  }

  // Write to this simd_view object.
  simd_view &write(const value_type &Val) {
    M_base.writeRegion(M_region, Val.data());
    return *this;
  }

  // whole region update with predicates
  void merge(const value_type &Val, const mask_type_t<length> &Mask) {
    merge(Val, read(), Mask);
  }

  void merge(const value_type &Val1, value_type Val2,
             const mask_type_t<length> &Mask) {
    Val2.merge(Val1, Mask);
    write(Val2.read());
  }

  // View this object in a different element type.
  template <typename EltTy> auto format() {
    using TopRegionTy = compute_format_type_t<simd_view, EltTy>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  // View as a 2-dimensional simd_view.
  template <typename EltTy, int Height, int Width> auto format() {
    using TopRegionTy =
        compute_format_type_2d_t<simd_view, EltTy, Height, Width>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(0, 0);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  // \brief 1D region select, apply a region on top of this object.
  //
  // @param Size the number of elements to be selected.
  //
  // @param Stride the element distance between two consecutive elements.
  //
  // @param Offset the starting element offset.
  //
  // @return the representing region object.
  //
  template <int Size, int Stride, typename T = simd_view,
            typename = std::enable_if_t<T::is1D()>>
  auto select(uint16_t Offset = 0) {
    using TopRegionTy = region1d_t<element_type, Size, Stride>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(Offset);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

  // \brief 2D region select, apply a region on top of this object.
  //
  // @param SizeX the number of elements to be selected in X-dimension.
  //
  // @param StrideX the element distance between two consecutive elements in
  // X-dimension.
  //
  // @param SizeY the number of elements to be selected in Y-dimension.
  //
  // @param StrideY the element distance between two consecutive elements in
  // Y-dimension.
  //
  // @param OffsetX the starting element offset in X-dimension.
  //
  // @param OffsetY the starting element offset in Y-dimension.
  //
  // @return the representing region object.
  //
  template <int SizeY, int StrideY, int SizeX, int StrideX,
            typename T = simd_view, typename = std::enable_if_t<T::is2D()>>
  auto select(uint16_t OffsetY = 0, uint16_t OffsetX = 0) {
    using TopRegionTy =
        region2d_t<element_type, SizeY, StrideY, SizeX, StrideX>;
    using NewRegionTy = std::pair<TopRegionTy, RegionTy>;
    using RetTy = simd_view<BaseTy, NewRegionTy>;
    TopRegionTy TopReg(OffsetY, OffsetX);
    return RetTy{this->M_base, std::make_pair(TopReg, M_region)};
  }

#define DEF_BINOP(BINOP, OPASSIGN)                                             \
  auto operator BINOP(const value_type &RHS) const {                           \
    using ComputeTy = compute_type_t<value_type>;                              \
    auto V0 = convert<typename ComputeTy::vector_type>(read().data());         \
    auto V1 = convert<typename ComputeTy::vector_type>(RHS.data());            \
    auto V2 = V0 BINOP V1;                                                     \
    return ComputeTy(V2);                                                      \
  }                                                                            \
  simd_view &operator OPASSIGN(const value_type &RHS) {                        \
    using ComputeTy = compute_type_t<value_type>;                              \
    auto V0 = convert<typename ComputeTy::vector_type>(read().data());         \
    auto V1 = convert<typename ComputeTy::vector_type>(RHS.data());            \
    auto V2 = V0 BINOP V1;                                                     \
    auto V3 = convert<vector_type>(V2);                                        \
    write(V3);                                                                 \
    return *this;                                                              \
  }

  DEF_BINOP(+, +=)
  DEF_BINOP(-, -=)
  DEF_BINOP(*, *=)
  DEF_BINOP(/, /=)

#undef DEF_BINOP

#define DEF_RELOP(RELOP)                                                       \
  simd<uint16_t, length> operator RELOP(const simd_view &RHS) const {          \
    auto R = read().data() RELOP RHS.read().data();                            \
    mask_type_t<length> M(1);                                                  \
    return M & convert<mask_type_t<length>>(R);                                \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP

#define DEF_LOGIC_OP(LOGIC_OP, OPASSIGN)                                       \
  simd_view operator LOGIC_OP(const simd_view &RHS) const {                    \
    static_assert(std::is_integral<element_type>(), "not integral type");      \
    auto V2 = read().data() LOGIC_OP RHS.read().data();                        \
    return simd_view(V2);                                                      \
  }                                                                            \
  simd_view &operator OPASSIGN(const simd_view &RHS) {                         \
    static_assert(std::is_integral<element_type>(), "not integeral type");     \
    auto V2 = read().data LOGIC_OP RHS.read().data();                          \
    auto V3 = convert<vector_type>(V2);                                        \
    write(V3);                                                                 \
    return *this;                                                              \
  }

  DEF_LOGIC_OP(&, &=)
  DEF_LOGIC_OP(|, |=)
  DEF_LOGIC_OP(^, ^=)

#undef DEF_LOGIC_OP

  // Operator ++, --
  simd_view &operator++() {
    *this += 1;
    return *this;
  }
  value_type operator++(int) {
    value_type Ret(read());
    operator++();
    return Ret;
  }
  simd_view &operator--() {
    *this -= 1;
    return *this;
  }
  value_type operator--(int) {
    value_type Ret(read());
    operator++();
    return Ret;
  }

  // Reference a row from a 2D region. This returns a 1D region.
  template <typename T = simd_view, typename = std::enable_if_t<T::is2D()>>
  auto row(int i) {
    return select<1, 0, getSizeX(), 1>(i, 0).template format<element_type>();
  }

  // Reference a column from a 2D region. This returns a 2D region.
  template <typename T = simd_view, typename = std::enable_if_t<T::is2D()>>
  auto column(int i) {
    return select<getSizeY(), 1, 1, 0>(0, i);
  }

  // Read a single element from a 1D region, by value only.
  template <typename T = simd_view, typename = std::enable_if_t<T::is1D()>>
  element_type operator[](int i) const {
    return read()[i];
  }

  // \brief replicate operation, replicate simd instance given a simd_view
  //
  // @param Rep number of times region has to be replicated
  //
  // @param OffsetX column offset in number of elements in src region
  //
  // @param OffsetY row offset in number of elements in src region
  //
  // @param VS vertical stride of src region to replicate
  //
  // @param W width of src region to replicate
  //
  // @param HS horizontal stride of src region to replicate
  //
  // @return replicated simd instance

  template <int Rep> simd<element_type, Rep> replicate() {
    return read().replicate<Rep>(0);
  }

  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return replicate<Rep, 0, W>(0, OffsetX);
  }

  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    return replicate<Rep, 0, W>(OffsetY, OffsetX);
  }

  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return replicate<Rep, VS, W, 1>(0, OffsetX);
  }

  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    return replicate<Rep, VS, W, 1>(OffsetY, OffsetX);
  }

  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX) {
    return read().template replicate<Rep, VS, W, HS>(OffsetX);
  }

  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY, uint16_t OffsetX) {
    constexpr int RowSize = is2D() ? getSizeX() : 0;
    return read().template replicate<Rep, VS, W, HS>(OffsetY * RowSize +
                                                     OffsetX);
  }

  // \brief any operation
  //
  // @return 1 if any element is set, 0 otherwise

  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return read().any();
  }

  // \brief all operation
  //
  // @return 1 if all elements are set, 0 otherwise

  template <typename T1 = element_type, typename T2 = BaseTy,
            typename = std::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return read().all();
  }

private:
  // The reference to the base object, which must be a simd object
  BaseTy &M_base;

  // The region applied on the base object. Its type could be
  // - region1d_t
  // - region2d_t
  // - std::pair<top_region_type, base_region_type>
  //
  RegionTy M_region;
};

} // namespace gpu
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
