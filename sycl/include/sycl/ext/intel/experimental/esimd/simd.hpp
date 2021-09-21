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

#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

#include <sycl/ext/intel/experimental/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

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
template <typename Ty, int N> class simd {
  template <typename, typename> friend class simd_view;

public:
  /// The underlying builtin data type.
  using vector_type = detail::vector_type_t<Ty, N>;

  /// The element type of this simd object.
  using element_type = Ty;

  /// The number of elements in this simd object.
  static constexpr int length = N;

  /// @{
  /// Constructors.
  simd() = default;
  simd(const simd &other) { set(other.data()); }
  template <typename SrcTy> simd(const simd<SrcTy, N> &other) {
    set(__builtin_convertvector(other.data(), detail::vector_type_t<Ty, N>));
  }
  template <typename SrcTy> simd(simd<SrcTy, N> &&other) {
    if constexpr (std::is_same<SrcTy, Ty>::value)
      set(other.data());
    else
      set(__builtin_convertvector(other.data(), detail::vector_type_t<Ty, N>));
  }
  simd(const vector_type &Val) { set(Val); }

  // TODO @rolandschulz
  // {quote}
  // Providing both an overload of initializer-list and the same type itself
  // causes really weird behavior. E.g.
  //   simd s1(1,2); //calls next constructor
  //   simd s2{1,2}; //calls this constructor
  // This might not be confusing for all users but to everyone using
  // uniform-initialization syntax. Therefore if you want to use this
  // constructor the other one should have a special type (see
  // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es64-use-the-tenotation-for-construction)
  // to avoid this issue. Also this seems like one of those areas where this
  // simd-type needless differs from std::simd. Why should these constructors be
  // different? Why reinvent the wheel and have all the work of fixing these
  // problems if we could just use the existing solution. Especially if that is
  // anyhow the long-term goal. Adding extra stuff like the select is totally
  // fine. But differ on things which have no apparent advantage and aren't as
  // thought through seems to have only downsides.
  // {/quote}

  simd(std::initializer_list<Ty> Ilist) noexcept {
    int i = 0;
    for (auto It = Ilist.begin(); It != Ilist.end() && i < N; ++It) {
      M_data[i++] = *It;
    }
  }

  /// Initialize a simd with an initial value and step.
  simd(Ty Val, Ty Step = Ty()) noexcept {
    if (Step == Ty())
      M_data = Val;
    else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        M_data[i] = Val;
        Val += Step;
      }
    }
  }
  /// @}

  /// conversion operator
  operator const vector_type &() const & { return M_data; }
  operator vector_type &() & { return M_data; }

  /// Implicit conversion for simd<T, 1> into T.
  template <typename T = simd,
            typename = sycl::detail::enable_if_t<T::length == 1>>
  operator element_type() const {
    return data()[0];
  }

  vector_type data() const {
#ifndef __SYCL_DEVICE_ONLY__
    return M_data;
#else
    return __esimd_vload<Ty, N>(&M_data);
#endif
  }

  /// Whole region read.
  simd read() const { return data(); }

  /// Whole region write.
  simd &write(const simd &Val) {
    set(Val.data());
    return *this;
  }

  /// Whole region update with predicates.
  void merge(const simd &Val, const mask_type_t<N> &Mask) {
    set(__esimd_wrregion<element_type, N, N, 0 /*VS*/, N, 1, N>(
        data(), Val.data(), 0, Mask));
  }
  void merge(const simd &Val1, simd Val2, const mask_type_t<N> &Mask) {
    Val2.merge(Val1, Mask);
    set(Val2.data());
  }

  /// View this simd object in a different element type.
  template <typename EltTy> auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy = detail::compute_format_type_t<simd, EltTy>;
    using RetTy = simd_view<simd, TopRegionTy>;
    TopRegionTy R(0);
    return RetTy{*this, R};
  }

  template <typename EltTy>
  __SYCL_DEPRECATED("use simd::bit_cast_view.")
  auto format() & {
    return bit_cast_view<EltTy>();
  }

  /// View as a 2-dimensional simd_view.
  template <typename EltTy, int Height, int Width>
  auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy =
        detail::compute_format_type_2d_t<simd, EltTy, Height, Width>;
    using RetTy = simd_view<simd, TopRegionTy>;
    TopRegionTy R(0, 0);
    return RetTy{*this, R};
  }

  template <typename EltTy, int Height, int Width>
  __SYCL_DEPRECATED("use simd::bit_cast_view.")
  auto format() & {
    return bit_cast_view<EltTy, Height, Width>();
  }

  /// 1D region select, apply a region on top of this LValue object.
  ///
  /// \tparam Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the representing region object.
  template <int Size, int Stride>
  simd_view<simd, region1d_t<Ty, Size, Stride>> select(uint16_t Offset = 0) &[
      [clang::lifetimebound]] {
    region1d_t<Ty, Size, Stride> Reg(Offset);
    return {*this, Reg};
  }

  /// 1D region select, apply a region on top of this RValue object.
  ///
  /// \tparam Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the value this region object refers to.
  template <int Size, int Stride>
  simd<Ty, Size> select(uint16_t Offset = 0) && {
    simd<Ty, N> &&Val = *this;
    return __esimd_rdregion<Ty, N, Size, /*VS*/ 0, Size, Stride>(Val.data(),
                                                                 Offset);
  }

  /// Read single element, return value only (not reference).
  Ty operator[](int i) const { return data()[i]; }

  /// Read single element, return value only (not reference).
  __SYCL_DEPRECATED("use operator[] form.")
  Ty operator()(int i) const { return data()[i]; }

  /// Return writable view of a single element.
  simd_view<simd, region1d_t<Ty, 1, 0>> operator[](int i)
      [[clang::lifetimebound]] {
    return select<1, 0>(i);
  }

  /// Return writable view of a single element.
  __SYCL_DEPRECATED("use operator[] form.")
  simd_view<simd, region1d_t<Ty, 1, 0>> operator()(int i) {
    return select<1, 0>(i);
  }

  // TODO ESIMD_EXPERIMENTAL
  /// Read multiple elements by their indices in vector
  template <int Size>
  simd<Ty, Size> iselect(const simd<uint16_t, Size> &Indices) {
    detail::vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(Ty);
    return __esimd_rdindirect<Ty, N, Size>(data(), Offsets);
  }
  // TODO ESIMD_EXPERIMENTAL
  /// update single element
  void iupdate(ushort Index, Ty V) {
    auto Val = data();
    Val[Index] = V;
    set(Val);
  }
  // TODO ESIMD_EXPERIMENTAL
  /// update multiple elements by their indices in vector
  template <int Size>
  void iupdate(const simd<uint16_t, Size> &Indices, const simd<Ty, Size> &Val,
               mask_type_t<Size> Mask) {
    detail::vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(Ty);
    set(__esimd_wrindirect<Ty, N, Size>(data(), Val.data(), Offsets, Mask));
  }

  // Use auto as a return type to honor C++ integer promotion rules,
  // e.g. simd<short,4>  +  simd<short,4>  ->  simd<int,4>
#define DEF_BINOP(BINOP, OPASSIGN)                                             \
  ESIMD_INLINE friend auto operator BINOP(const simd &X, const simd &Y) {      \
    using ComputeTy = detail::compute_type_t<simd>;                            \
    auto V0 = detail::convert<typename ComputeTy::vector_type>(X.data());      \
    auto V1 = detail::convert<typename ComputeTy::vector_type>(Y.data());      \
    auto V2 = V0 BINOP V1;                                                     \
    return ComputeTy(V2);                                                      \
  }                                                                            \
  template <typename T1, typename T = simd,                                    \
            typename = sycl::detail::enable_if_t<T::length == 1 &&             \
                                                 std::is_arithmetic_v<T1>>>    \
  ESIMD_INLINE friend auto operator BINOP(const simd &X, T1 Y) {               \
    return X BINOP simd((Ty)Y);                                                \
  }                                                                            \
  ESIMD_INLINE friend simd &operator OPASSIGN(simd &LHS, const simd &RHS) {    \
    using ComputeTy = detail::compute_type_t<simd>;                            \
    auto V0 = detail::convert<typename ComputeTy::vector_type>(LHS.data());    \
    auto V1 = detail::convert<typename ComputeTy::vector_type>(RHS.data());    \
    auto V2 = V0 BINOP V1;                                                     \
    LHS.write(detail::convert<vector_type>(V2));                               \
    return LHS;                                                                \
  }                                                                            \
  ESIMD_INLINE friend simd &operator OPASSIGN(simd &LHS, const Ty &RHS) {      \
    LHS OPASSIGN simd(RHS);                                                    \
    return LHS;                                                                \
  }

  DEF_BINOP(+, +=)
  DEF_BINOP(-, -=)
  DEF_BINOP(*, *=)
  DEF_BINOP(/, /=)
  DEF_BINOP(%, %=)

#undef DEF_BINOP

  // TODO @rolandschulz, @mattkretz
  // Introduce simd_mask type and let user use this type instead of specific
  // type representation (simd<uint16_t, N>) to make it more portable
  // TODO @iburyl should be mask_type_t, which might become more abstracted in
  // the future revisions.
  //
#define DEF_RELOP(RELOP)                                                       \
  ESIMD_INLINE friend simd<uint16_t, N> operator RELOP(const simd &X,          \
                                                       const simd &Y) {        \
    auto R = X.data() RELOP Y.data();                                          \
    mask_type_t<N> M(1);                                                       \
    return M & detail::convert<mask_type_t<N>>(R);                             \
  }                                                                            \
  template <typename T1, typename T = simd,                                    \
            typename = sycl::detail::enable_if_t<(T::length == 1) &&           \
                                                 std::is_arithmetic_v<T1>>>    \
  ESIMD_INLINE friend bool operator RELOP(const simd &X, T1 Y) {               \
    return (Ty)X RELOP(Ty) Y;                                                  \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP

#define DEF_BITWISE_OP(BITWISE_OP, OPASSIGN)                                   \
  ESIMD_INLINE friend simd operator BITWISE_OP(const simd &X, const simd &Y) { \
    static_assert(std::is_integral<Ty>(), "not integeral type");               \
    auto V2 = X.data() BITWISE_OP Y.data();                                    \
    return simd(V2);                                                           \
  }                                                                            \
  ESIMD_INLINE friend simd &operator OPASSIGN(simd &LHS, const simd &RHS) {    \
    static_assert(std::is_integral<Ty>(), "not integeral type");               \
    auto V2 = LHS.data() BITWISE_OP RHS.data();                                \
    LHS.write(detail::convert<vector_type>(V2));                               \
    return LHS;                                                                \
  }                                                                            \
  ESIMD_INLINE friend simd &operator OPASSIGN(simd &LHS, const Ty &RHS) {      \
    LHS OPASSIGN simd(RHS);                                                    \
    return LHS;                                                                \
  }

  DEF_BITWISE_OP(&, &=)
  DEF_BITWISE_OP(|, |=)
  DEF_BITWISE_OP(^, ^=)
  DEF_BITWISE_OP(<<, <<=)
  DEF_BITWISE_OP(>>, >>=)

#undef DEF_BITWISE_OP

  // Operator ++, --
  simd &operator++() {
    *this += 1;
    return *this;
  }
  simd operator++(int) {
    simd Ret(*this);
    operator++();
    return Ret;
  }
  simd &operator--() {
    *this -= 1;
    return *this;
  }
  simd operator--(int) {
    simd Ret(*this);
    operator--();
    return Ret;
  }

#define DEF_UNARY_OP(UNARY_OP)                                                 \
  simd operator UNARY_OP() {                                                   \
    auto V = UNARY_OP(data());                                                 \
    return simd(V);                                                            \
  }
  DEF_UNARY_OP(~)
  DEF_UNARY_OP(+)
  DEF_UNARY_OP(-)

#undef DEF_UNARY_OP

  // negation operator
  auto operator!() { return *this == 0; }

  /// \name Replicate
  /// Replicate simd instance given a region.
  /// @{
  ///

  /// \tparam Rep is number of times region has to be replicated.
  /// \return replicated simd instance.
  template <int Rep> simd<Ty, Rep * N> replicate() {
    return replicate<Rep, N>(0);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W>
  __SYCL_DEPRECATED("use simd::replicate_w")
  simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return replicate_w<Rep, W>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W> simd<Ty, Rep * W> replicate_w(uint16_t Offset) {
    return replicate_vs_w_hs<Rep, 0, W, 1>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  __SYCL_DEPRECATED("use simd::replicate_vs_w")
  simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return replicate_vs_w<Rep, VS, W>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W width of src region to replicate.
  /// \param Offset offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  simd<Ty, Rep * W> replicate_vs_w(uint16_t Offset) {
    return replicate_vs_w_hs<Rep, VS, W, 1>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS horizontal stride of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  __SYCL_DEPRECATED("use simd::replicate_vs_w_hs")
  simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return replicate_vs_w_hs<Rep, VS, W, HS>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS horizontal stride of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  simd<Ty, Rep * W> replicate_vs_w_hs(uint16_t Offset) {
    return __esimd_rdregion<element_type, N, Rep * W, VS, W, HS, N>(
        data(), Offset * sizeof(Ty));
  }
  ///@}

  /// Any operation.
  ///
  /// \return 1 if any element is set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = Ty,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return __esimd_any<Ty, N>(data());
  }

  /// All operation.
  ///
  /// \return 1 if all elements are set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = Ty,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return __esimd_all<Ty, N>(data());
  }

  /// Write a simd-vector into a basic region of a simd object.
  template <typename RTy>
  ESIMD_INLINE void
  writeRegion(RTy Region,
              const detail::vector_type_t<typename RTy::element_type,
                                          RTy::length> &Val) {
    using ElemTy = typename RTy::element_type;
    if constexpr (N * sizeof(Ty) == RTy::length * sizeof(ElemTy))
      // update the entire vector
      set(detail::bitcast<Ty, ElemTy, RTy::length>(Val));
    else {
      static_assert(!RTy::Is_2D);
      // If element type differs, do bitcast conversion first.
      auto Base = detail::bitcast<ElemTy, Ty, N>(data());
      constexpr int BN = (N * sizeof(Ty)) / sizeof(ElemTy);
      // Access the region information.
      constexpr int M = RTy::Size_x;
      constexpr int Stride = RTy::Stride_x;
      uint16_t Offset = Region.M_offset_x * sizeof(ElemTy);

      // Merge and update.
      auto Merged = __esimd_wrregion<ElemTy, BN, M,
                                     /*VS*/ 0, M, Stride>(Base, Val, Offset);
      // Convert back to the original element type, if needed.
      set(detail::bitcast<Ty, ElemTy, BN>(Merged));
    }
  }

  /// Write a simd-vector into a nested region of a simd object.
  template <typename TR, typename UR>
  ESIMD_INLINE void writeRegion(
      std::pair<TR, UR> Region,
      const detail::vector_type_t<typename TR::element_type, TR::length> &Val) {
    // parent-region type
    using PaTy = typename shape_type<UR>::type;
    using ElemTy = typename TR::element_type;
    using BT = typename PaTy::element_type;
    constexpr int BN = PaTy::length;

    if constexpr (PaTy::Size_in_bytes == TR::Size_in_bytes) {
      writeRegion(Region.second, detail::bitcast<BT, ElemTy, TR::length>(Val));
    } else {
      // Recursively read the base
      auto Base = detail::readRegion<Ty, N>(data(), Region.second);
      // If element type differs, do bitcast conversion first.
      auto Base1 = detail::bitcast<ElemTy, BT, BN>(Base);
      constexpr int BN1 = PaTy::Size_in_bytes / sizeof(ElemTy);

      if constexpr (!TR::Is_2D) {
        // Access the region information.
        constexpr int M = TR::Size_x;
        constexpr int Stride = TR::Stride_x;
        uint16_t Offset = Region.first.M_offset_x * sizeof(ElemTy);

        // Merge and update.
        Base1 = __esimd_wrregion<ElemTy, BN1, M,
                                 /*VS*/ 0, M, Stride>(Base1, Val, Offset);
      } else {
        static_assert(std::is_same<ElemTy, BT>::value);
        // Read columns with non-trivial horizontal stride.
        constexpr int M = TR::length;
        constexpr int VS = PaTy::Size_x * TR::Stride_y;
        constexpr int W = TR::Size_x;
        constexpr int HS = TR::Stride_x;
        constexpr int ParentWidth = PaTy::Size_x;

        // Compute the byte offset for the starting element.
        uint16_t Offset = static_cast<uint16_t>(
            (Region.first.M_offset_y * PaTy::Size_x + Region.first.M_offset_x) *
            sizeof(ElemTy));

        // Merge and update.
        Base1 = __esimd_wrregion<ElemTy, BN1, M, VS, W, HS, ParentWidth>(
            Base1, Val, Offset);
      }
      // Convert back to the original element type, if needed.
      auto Merged1 = detail::bitcast<BT, ElemTy, BN1>(Base1);
      // recursively write it back to the base
      writeRegion(Region.second, Merged1);
    }
  }

  /// @name Memory operations
  /// TODO NOTE: These APIs do not support cache hint specification yet, as this
  /// is WIP. Later addition of hints is not expected to break code using these
  /// APIs.
  ///
  /// @{

  /// Copy a contiguous block of data from memory into this simd object.
  /// The amount of memory copied equals the total size of vector elements in
  /// this object.
  /// @param addr the memory address to copy from. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  ESIMD_INLINE void copy_from(const Ty *const addr) SYCL_ESIMD_FUNCTION;

  /// Copy a contiguous block of data from memory into this simd object.
  /// The amount of memory copied equals the total size of vector elements in
  /// this object.
  /// Source memory location is represented via a global accessor and offset.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from.
  template <typename AccessorT>
  ESIMD_INLINE
      detail::EnableIfAccessor<AccessorT, detail::accessor_mode_cap::can_read,
                               sycl::access::target::global_buffer, void>
      copy_from(AccessorT acc, uint32_t offset) SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// @param addr the memory address to copy to. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  ESIMD_INLINE void copy_to(Ty *addr) SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// Destination memory location is represented via a global accessor and
  /// offset.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from.
  template <typename AccessorT>
  ESIMD_INLINE
      detail::EnableIfAccessor<AccessorT, detail::accessor_mode_cap::can_write,
                               sycl::access::target::global_buffer, void>
      copy_to(AccessorT acc, uint32_t offset) SYCL_ESIMD_FUNCTION;

  /// @} // Memory operations
private:
  // The underlying data for this vector.
  vector_type M_data;

  void set(const vector_type &Val) {
#ifndef __SYCL_DEVICE_ONLY__
    M_data = Val;
#else
    __esimd_vstore<Ty, N>(&M_data, Val);
#endif
  }
};

template <typename U, typename T, int n>
ESIMD_INLINE simd<U, n> convert(simd<T, n> val) {
  return __builtin_convertvector(val.data(), detail::vector_type_t<U, n>);
}

// ----------- Outlined implementations of esimd class APIs.

template <typename T, int N> void simd<T, N>::copy_from(const T *const Addr) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t AddrVal = reinterpret_cast<uintptr_t>(Addr);
  *this =
      __esimd_flat_block_read_unaligned<T, N, CacheHint::None, CacheHint::None>(
          AddrVal);
}

template <typename T, int N>
template <typename AccessorT>
ESIMD_INLINE
    detail::EnableIfAccessor<AccessorT, detail::accessor_mode_cap::can_read,
                             sycl::access::target::global_buffer, void>
    simd<T, N>::copy_from(AccessorT acc, uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = detail::AccessorPrivateProxy::getNativeImageObj(acc);
  *this = __esimd_block_read<T, N>(surf_ind, offset);
#else
  *this = __esimd_block_read<T, N>(acc, offset);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename T, int N> void simd<T, N>::copy_to(T *addr) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t AddrVal = reinterpret_cast<uintptr_t>(addr);
  __esimd_flat_block_write<T, N, CacheHint::None, CacheHint::None>(AddrVal,
                                                                   data());
}

template <typename T, int N>
template <typename AccessorT>
ESIMD_INLINE
    detail::EnableIfAccessor<AccessorT, detail::accessor_mode_cap::can_write,
                             sycl::access::target::global_buffer, void>
    simd<T, N>::copy_to(AccessorT acc, uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = detail::AccessorPrivateProxy::getNativeImageObj(acc);
  __esimd_block_write<T, N>(surf_ind, offset >> 4, data());
#else
  __esimd_block_write<T, N>(acc, offset >> 4, data());
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifndef __SYCL_DEVICE_ONLY__
template <typename Ty, int N>
std::ostream &
operator<<(std::ostream &OS,
           const sycl::ext::intel::experimental::esimd::simd<Ty, N> &V) {
  OS << "{";
  for (int I = 0; I < N; I++) {
    OS << V[I];
    if (I < N - 1)
      OS << ",";
  }
  OS << "}";
  return OS;
}

#endif
