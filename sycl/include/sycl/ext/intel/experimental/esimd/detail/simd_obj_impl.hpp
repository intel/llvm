//==------------ - simd_obj_impl.hpp - DPC++ Explicit SIMD API -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD vector APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

/// The simd_obj_impl vector class.
///
/// This is a base class for all ESIMD simd classes with real storage (simd,
/// simd_mask_impl). It wraps a clang vector as the storage for the elements.
/// Additionally this class supports region operations that map to Intel GPU
/// regions. The type of a region select or bit_cast_view operation is of
/// simd_view type, which models read-update-write semantics.
///
/// For the is_simd_obj_impl_derivative helper to work correctly, all derived
/// classes must be templated by element type and number of elements. If fewer
/// template arguments are needed, template aliases can be used
/// (simd_mask_type).
///
/// \tparam Ty the element type
/// \tparam N number of elements
/// \tparam Derived - a class derived from this one; this class and its
///    derivatives must follow the 'curiously recurring template' pattern.
/// \tparam SFINAE - defaults to 'void' in the forward declarion within
///    types.hpp, used to disable invalid specializations.
///
/// \ingroup sycl_esimd
template <typename Ty, int N, class Derived, class SFINAE> class simd_obj_impl {
  template <typename, typename> friend class simd_view;
  template <typename, int> friend class simd;
  template <typename, int> friend class simd_mask_impl;

public:
  /// The underlying builtin data type.
  using vector_type = vector_type_t<Ty, N>;

  /// The element type of this simd_obj_impl object.
  using element_type = Ty;

  /// The number of elements in this simd_obj_impl object.
  static constexpr int length = N;

protected:
  template <int N1, class = std::enable_if_t<N1 == N>>
  void init_from_array(const Ty(&&Arr)[N1]) noexcept {
    for (auto I = 0; I < N; ++I) {
      M_data[I] = Arr[I];
    }
  }

private:
  Derived &cast_this_to_derived() { return reinterpret_cast<Derived &>(*this); }

public:
  /// @{
  /// Constructors.
  simd_obj_impl() = default;
  simd_obj_impl(const simd_obj_impl &other) {
    __esimd_dbg_print(simd_obj_impl(const simd_obj_impl &other));
    set(other.data());
  }

  /// Implicit conversion constructor from another \c simd_obj_impl object.
  template <typename SrcTy>
  simd_obj_impl(
      const simd_obj_impl<SrcTy, N, convert_simd_elem_type_t<Derived, SrcTy>,
                          SFINAE> &other) {
    __esimd_dbg_print(simd_obj_impl(const simd_obj_impl... > &other));
    if constexpr (std::is_same_v<SrcTy, Ty>)
      set(other.data());
    else
      set(__builtin_convertvector(other.data(), vector_type));
  }

  /// Implicit conversion constructor from a raw vector object.
  simd_obj_impl(const vector_type &Val) {
    __esimd_dbg_print(simd_obj_impl(const vector_type &Val));
    set(Val);
  }

  /// This constructor is deprecated for two reasons:
  /// 1) it adds confusion between
  ///   simd s1(1,2); //calls next constructor
  ///   simd s2{1,2}; //calls this constructor (uniform initialization syntax)
  /// 2) no compile-time control over the size of the initializer; e.g. the
  ///    following will compile:
  ///   simd<int, 2> x = {1, 2, 3, 4};
  __SYCL_DEPRECATED("use constructor from array, e.g: simd<int,3> x({1,2,3});")
  simd_obj_impl(std::initializer_list<Ty> Ilist) noexcept {
    __esimd_dbg_print(simd_obj_impl(std::initializer_list<Ty> Ilist));
    int i = 0;
    for (auto It = Ilist.begin(); It != Ilist.end() && i < N; ++It) {
      M_data[i++] = *It;
    }
  }

  /// Initialize a simd_obj_impl object with an initial value and step.
  simd_obj_impl(Ty Val, Ty Step) noexcept {
    __esimd_dbg_print(simd_obj_impl(Ty Val, Ty Step));
#pragma unroll
    for (int i = 0; i < N; ++i) {
      M_data[i] = Val;
      Val += Step;
    }
  }

  /// Broadcast constructor
  simd_obj_impl(Ty Val) noexcept {
    __esimd_dbg_print(simd_obj_impl(Ty Val));
    M_data = Val;
  }

  /// Construct from an array. To allow e.g. simd_mask_type<N> m({1,0,0,1,...}).
  template <int N1, class = std::enable_if_t<N1 == N>>
  simd_obj_impl(const Ty(&&Arr)[N1]) noexcept {
    __esimd_dbg_print(simd_obj_impl(const Ty(&&Arr)[N1]));
    init_from_array(std::move(Arr));
  }

  /// @}

  // Load the object's value from array.
  template <int N1> std::enable_if_t<N1 == N> copy_from(const Ty (&Arr)[N1]) {
    __esimd_dbg_print(copy_from(const Ty(&Arr)[N1]));
    vector_type Tmp;
    for (auto I = 0; I < N; ++I) {
      Tmp[I] = Arr[I];
    }
    set(Tmp);
  }

  // Store the object's value to array.
  template <int N1> std::enable_if_t<N1 == N> copy_to(Ty (&Arr)[N1]) const {
    __esimd_dbg_print(copy_to(Ty(&Arr)[N1]));
    for (auto I = 0; I < N; ++I) {
      Arr[I] = data()[I];
    }
  }

  /// @{
  /// Conversion operators.
  explicit operator const vector_type &() const & {
    __esimd_dbg_print(explicit operator const vector_type &() const &);
    return M_data;
  }
  explicit operator vector_type &() & {
    __esimd_dbg_print(explicit operator vector_type &() &);
    return M_data;
  }

  /// Explicit conversion for simd_obj_impl<T, 1> into T.
  template <typename T = simd_obj_impl,
            typename = sycl::detail::enable_if_t<T::length == 1>>
  operator Ty() const {
    __esimd_dbg_print(explicit operator Ty());
    return data()[0];
  }
  /// @}

  vector_type data() const {
    __esimd_dbg_print(vector_type data());
#ifndef __SYCL_DEVICE_ONLY__
    return M_data;
#else
    return __esimd_vload<Ty, N>(&M_data);
#endif
  }

  /// Whole region read.
  Derived read() const { return Derived{data()}; }

  /// Whole region write.
  Derived &write(const Derived &Val) {
    set(Val.data());
    return cast_this_to_derived();
  }

  /// Whole region update with predicates.
  void merge(const Derived &Val, const simd_mask_type<N> &Mask) {
    set(__esimd_wrregion<Ty, N, N, 0 /*VS*/, N, 1, N>(data(), Val.data(), 0,
                                                      Mask.data()));
  }

  void merge(const Derived &Val1, Derived Val2, const simd_mask_type<N> &Mask) {
    Val2.merge(Val1, Mask);
    set(Val2.data());
  }

  /// View this simd_obj_impl object in a different element type.
  template <typename EltTy> auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy = compute_format_type_t<simd_obj_impl, EltTy>;
    using RetTy = simd_view<Derived, TopRegionTy>;
    return RetTy{cast_this_to_derived(), TopRegionTy{0}};
  }

  template <typename EltTy>
  __SYCL_DEPRECATED("use simd_obj_impl::bit_cast_view.")
  auto format() & {
    return bit_cast_view<EltTy>();
  }

  /// View as a 2-dimensional simd_view.
  template <typename EltTy, int Height, int Width>
  auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy =
        compute_format_type_2d_t<simd_obj_impl, EltTy, Height, Width>;
    using RetTy = simd_view<Derived, TopRegionTy>;
    return RetTy{cast_this_to_derived(), TopRegionTy{0, 0}};
  }

  template <typename EltTy, int Height, int Width>
  __SYCL_DEPRECATED("use simd_obj_impl::bit_cast_view.")
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
  simd_view<Derived, region1d_t<Ty, Size, Stride>>
  select(uint16_t Offset = 0) &[[clang::lifetimebound]] {
    static_assert(Size > 1 || Stride == 1,
                  "Stride must be 1 in single-element region");
    region1d_t<Ty, Size, Stride> Reg(Offset);
    return {cast_this_to_derived(), std::move(Reg)};
  }

  /// 1D region select, apply a region on top of this RValue object.
  ///
  /// \tparam Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the value this region object refers to.
  template <int Size, int Stride>
  resize_a_simd_type_t<Derived, Size> select(uint16_t Offset = 0) && {
    static_assert(Size > 1 || Stride == 1,
                  "Stride must be 1 in single-element region");
    Derived &&Val = std::move(cast_this_to_derived());
    return __esimd_rdregion<Ty, N, Size, /*VS*/ 0, Size, Stride>(Val.data(),
                                                                 Offset);
  }

  /// Read single element, return value only (not reference).
  Ty operator[](int i) const { return data()[i]; }

  /// Read single element, return value only (not reference).
  __SYCL_DEPRECATED("use operator[] form.")
  Ty operator()(int i) const { return data()[i]; }

  /// Return writable view of a single element.
  simd_view<Derived, region1d_scalar_t<Ty>> operator[](int i)
      [[clang::lifetimebound]] {
    return select<1, 1>(i);
  }

  /// Return writable view of a single element.
  __SYCL_DEPRECATED("use operator[] form.")
  simd_view<Derived, region1d_scalar_t<Ty>> operator()(int i) {
    return select<1, 1>(i);
  }

  // TODO ESIMD_EXPERIMENTAL
  /// Read multiple elements by their indices in vector
  template <int Size>
  resize_a_simd_type_t<Derived, Size>
  iselect(const simd<uint16_t, Size> &Indices) {
    vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(Ty);
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
  void iupdate(const simd<uint16_t, Size> &Indices,
               const resize_a_simd_type_t<Derived, Size> &Val,
               const simd_mask_type<Size> &Mask) {
    vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(Ty);
    set(__esimd_wrindirect<Ty, N, Size>(data(), Val.data(), Offsets,
                                        Mask.data()));
  }

  /// \name Replicate
  /// Replicate simd_obj_impl instance given a region.
  /// @{
  ///

  /// \tparam Rep is number of times region has to be replicated.
  /// \return replicated simd_obj_impl instance.
  template <int Rep> resize_a_simd_type_t<Derived, Rep * N> replicate() const {
    return replicate<Rep, N>(0);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int W>
  __SYCL_DEPRECATED("use simd_obj_impl::replicate_w")
  resize_a_simd_type_t<Derived, Rep * W> replicate(uint16_t Offset) const {
    return replicate_w<Rep, W>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int W>
  resize_a_simd_type_t<Derived, Rep * W> replicate_w(uint16_t Offset) const {
    return replicate_vs_w_hs<Rep, 0, W, 1>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int VS, int W>
  __SYCL_DEPRECATED("use simd_obj_impl::replicate_vs_w")
  resize_a_simd_type_t<Derived, Rep * W> replicate(uint16_t Offset) const {
    return replicate_vs_w<Rep, VS, W>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W width of src region to replicate.
  /// \param Offset offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int VS, int W>
  resize_a_simd_type_t<Derived, Rep * W> replicate_vs_w(uint16_t Offset) const {
    return replicate_vs_w_hs<Rep, VS, W, 1>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS horizontal stride of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int VS, int W, int HS>
  __SYCL_DEPRECATED("use simd_obj_impl::replicate_vs_w_hs")
  resize_a_simd_type_t<Derived, Rep * W> replicate(uint16_t Offset) const {
    return replicate_vs_w_hs<Rep, VS, W, HS>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS horizontal stride of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd_obj_impl instance.
  template <int Rep, int VS, int W, int HS>
  resize_a_simd_type_t<Derived, Rep * W>
  replicate_vs_w_hs(uint16_t Offset) const {
    return __esimd_rdregion<Ty, N, Rep * W, VS, W, HS, N>(data(),
                                                          Offset * sizeof(Ty));
  }
  ///@}

  /// Any operation.
  ///
  /// \return 1 if any element is set, 0 otherwise.
  template <typename T1 = Ty,
            typename = std::enable_if_t<std::is_integral<T1>::value>>
  uint16_t any() const {
    return __esimd_any<Ty, N>(data());
  }

  /// All operation.
  ///
  /// \return 1 if all elements are set, 0 otherwise.
  template <typename T1 = Ty,
            typename = std::enable_if_t<std::is_integral<T1>::value>>
  uint16_t all() const {
    return __esimd_all<Ty, N>(data());
  }

  /// Write a simd_obj_impl-vector into a basic region of a simd_obj_impl
  /// object.
  template <typename RTy>
  ESIMD_INLINE void writeRegion(
      RTy Region,
      const vector_type_t<typename RTy::element_type, RTy::length> &Val) {
    using ElemTy = typename RTy::element_type;
    if constexpr (N * sizeof(Ty) == RTy::length * sizeof(ElemTy))
      // update the entire vector
      set(bitcast<Ty, ElemTy, RTy::length>(Val));
    else {
      static_assert(!RTy::Is_2D);
      // If element type differs, do bitcast conversion first.
      auto Base = bitcast<ElemTy, Ty, N>(data());
      constexpr int BN = (N * sizeof(Ty)) / sizeof(ElemTy);
      // Access the region information.
      constexpr int M = RTy::Size_x;
      constexpr int Stride = RTy::Stride_x;
      uint16_t Offset = Region.M_offset_x * sizeof(ElemTy);

      // Merge and update.
      auto Merged = __esimd_wrregion<ElemTy, BN, M,
                                     /*VS*/ 0, M, Stride>(Base, Val, Offset);
      // Convert back to the original element type, if needed.
      set(bitcast<Ty, ElemTy, BN>(Merged));
    }
  }

  /// Write a simd_obj_impl-vector into a nested region of a simd_obj_impl
  /// object.
  template <typename TR, typename UR>
  ESIMD_INLINE void
  writeRegion(std::pair<TR, UR> Region,
              const vector_type_t<typename TR::element_type, TR::length> &Val) {
    // parent-region type
    using PaTy = typename shape_type<UR>::type;
    using ElemTy = typename TR::element_type;
    using BT = typename PaTy::element_type;
    constexpr int BN = PaTy::length;

    if constexpr (PaTy::Size_in_bytes == TR::Size_in_bytes) {
      writeRegion(Region.second, bitcast<BT, ElemTy, TR::length>(Val));
    } else {
      // Recursively read the base
      auto Base = readRegion<Ty, N>(data(), Region.second);
      // If element type differs, do bitcast conversion first.
      auto Base1 = bitcast<ElemTy, BT, BN>(Base);
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
      auto Merged1 = bitcast<BT, ElemTy, BN1>(Base1);
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

  /// Copy a contiguous block of data from memory into this simd_obj_impl
  /// object. The amount of memory copied equals the total size of vector
  /// elements in this object.
  /// @param addr the memory address to copy from. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  ESIMD_INLINE void copy_from(const Ty *addr) SYCL_ESIMD_FUNCTION;

  /// Copy a contiguous block of data from memory into this simd_obj_impl
  /// object. The amount of memory copied equals the total size of vector
  /// elements in this object. Source memory location is represented via a
  /// global accessor and offset.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from (in bytes).
  template <typename AccessorT>
  ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_read,
                                sycl::access::target::global_buffer, void>
  copy_from(AccessorT acc, uint32_t offset) SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// @param addr the memory address to copy to. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  ESIMD_INLINE void copy_to(Ty *addr) const SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// Destination memory location is represented via a global accessor and
  /// offset.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from.
  template <typename AccessorT>
  ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_write,
                                sycl::access::target::global_buffer, void>
  copy_to(AccessorT acc, uint32_t offset) const SYCL_ESIMD_FUNCTION;

  /// @} // Memory operations

  /// Bitwise inversion, available in all subclasses.
  template <class T1 = Ty, class = std::enable_if_t<std::is_integral_v<T1>>>
  Derived operator~() const {
    return Derived(~data());
  }

  /// Unary logical negation operator, available in all subclasses.
  /// Similarly to C++, where !x returns bool, !simd returns as simd_mask, where
  /// each element is a result of comparision with zero.
  template <class T1 = Ty, class = std::enable_if_t<std::is_integral_v<T1>>>
  simd_mask_type<N> operator!() const {
    using MaskVecT = typename simd_mask_type<N>::vector_type;
    auto R = data() == vector_type(0);
    return simd_mask_type<N>{__builtin_convertvector(R, MaskVecT) &
                             MaskVecT(1)};
  }

#define __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(BINOP, OPASSIGN, COND)              \
                                                                               \
  /*  OPASSIGN simd_obj_impl */                                                \
  template <class T1, class SimdT,                                             \
            class = std::enable_if_t<(is_simd_type_v<Derived> ==               \
                                      is_simd_type_v<SimdT>)&&COND>>           \
  Derived &operator OPASSIGN(                                                  \
      const __SEIEED::simd_obj_impl<T1, N, SimdT> &RHS) {                      \
    auto Res = *this BINOP RHS;                                                \
    set(__SEIEED::convert<vector_type>(Res.data()));                           \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /*  OPASSIGN simd_view */                                                    \
  template <class SimdT1, class RegionT1,                                      \
            class T1 = typename RegionT1::element_type,                        \
            class = std::enable_if_t<                                          \
                (is_simd_type_v<Derived> ==                                    \
                 is_simd_type_v<SimdT1>)&&(RegionT1::length == length) &&      \
                COND>>                                                         \
  Derived &operator OPASSIGN(                                                  \
      const __SEIEE::simd_view<SimdT1, RegionT1> &RHS) {                       \
    auto Res = *this BINOP RHS.read();                                         \
    set(__SEIEED::convert<vector_type>(Res.data()));                           \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /*  OPASSIGN SCALAR */                                                       \
  template <class T1, class = std::enable_if_t<COND>>                          \
  Derived &operator OPASSIGN(T1 RHS) {                                         \
    if constexpr (is_simd_type_v<Derived>) {                                   \
      using RHSVecT = __SEIEED::construct_a_simd_type_t<Derived, T1, N>;       \
      return *this OPASSIGN RHSVecT(RHS);                                      \
    } else {                                                                   \
      return *this OPASSIGN Derived((Ty)RHS);                                  \
    }                                                                          \
  }

// Bitwise operations are defined for simd objects and masks, and both operands
// must be integral
#define __ESIMD_BITWISE_OP_FILTER                                              \
  std::is_integral_v<element_type> &&std::is_integral_v<T1>

  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(^, ^=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(|, |=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(&, &=, __ESIMD_BITWISE_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(%, %=, __ESIMD_BITWISE_OP_FILTER)
#undef __ESIMD_BITWISE_OP_FILTER

// Bit shift operations are defined only for simd objects (not for masks), and
// both operands must be integral
#define __ESIMD_SHIFT_OP_FILTER                                                \
  std::is_integral_v<element_type> &&std::is_integral_v<T1>                    \
      &&__SEIEED::is_simd_type_v<Derived>

  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(<<, <<=, __ESIMD_SHIFT_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(>>, >>=, __ESIMD_SHIFT_OP_FILTER)
#undef __ESIMD_SHIFT_OP_FILTER

// Arithmetic operations are defined only for simd objects, and the second
// operand's element type must be vectorizable. This requirement for 'this'
// is fulfilled, because otherwise 'this' couldn't have been constructed.
#define __ESIMD_ARITH_OP_FILTER                                                \
  __SEIEED::is_simd_type_v<Derived> &&__SEIEED::is_vectorizable_v<T1>

  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(+, +=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(-, -=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(*, *=, __ESIMD_ARITH_OP_FILTER)
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(/, /=, __ESIMD_ARITH_OP_FILTER)
#undef __ESIMD_ARITH_OP_FILTER

private:
  // The underlying data for this vector.
  vector_type M_data;

protected:
  void set(const vector_type &Val) {
#ifndef __SYCL_DEVICE_ONLY__
    M_data = Val;
#else
    __esimd_vstore<Ty, N>(&M_data, Val);
#endif
  }
};

// ----------- Outlined implementations of simd_obj_impl class APIs.

template <typename T, int N, class T1, class SFINAE>
void simd_obj_impl<T, N, T1, SFINAE>::copy_from(const T *Addr)
    SYCL_ESIMD_FUNCTION {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(isPowerOf2(Sz / OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t AddrVal = reinterpret_cast<uintptr_t>(Addr);
  *this = __esimd_svm_block_ld_unaligned<T, N>(AddrVal);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT>
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_read,
                              sycl::access::target::global_buffer, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc, uint32_t offset)
    SYCL_ESIMD_FUNCTION {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(isPowerOf2(Sz / OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * OperandSize::OWORD,
                "block size must be at most 8 owords");
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind =
      __esimd_get_surface_index(AccessorPrivateProxy::getNativeImageObj(acc));
  *this = __esimd_oword_ld_unaligned<T, N>(surf_ind, offset);
#else
  *this = __esimd_oword_ld_unaligned<T, N>(acc, offset);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename T, int N, class T1, class SFINAE>
void simd_obj_impl<T, N, T1, SFINAE>::copy_to(T *addr) const
    SYCL_ESIMD_FUNCTION {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(isPowerOf2(Sz / OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t AddrVal = reinterpret_cast<uintptr_t>(addr);
  __esimd_svm_block_st<T, N>(AddrVal, data());
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT>
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_write,
                              sycl::access::target::global_buffer, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc, uint32_t offset) const
    SYCL_ESIMD_FUNCTION {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(isPowerOf2(Sz / OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * OperandSize::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind =
      __esimd_get_surface_index(AccessorPrivateProxy::getNativeImageObj(acc));
  __esimd_oword_st<T, N>(surf_ind, offset >> 4, data());
#else
  __esimd_oword_st<T, N>(acc, offset >> 4, data());
#endif // __SYCL_DEVICE_ONLY__
}
} // namespace detail

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
