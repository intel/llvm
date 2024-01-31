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

#include <sycl/aspects.hpp>
#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/intrin.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/esimd/detail/sycl_util.hpp>
#include <sycl/ext/intel/esimd/detail/test_proxy.hpp>
#include <sycl/ext/intel/esimd/detail/type_format.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

/// @addtogroup sycl_esimd_core
/// @{

/// @defgroup sycl_esimd_core_align Alignment control.
/// Alignment type tags and related APIs for use with ESIMD memory access
/// operations. The basic restrictions for memory location specified as
/// parameters for memory access APIs supporting alignment control are as
/// follows:
/// - If alignment control parameter is \c element_aligned_tag, then the
///   location must be aligned by <code>alignof(T)</code>, where \c T is element
///   type.
/// - If it is \c vector_aligned_tag, the location must be aligned by
///   <code>alignof(VT)</code>,  where \c VT is the raw vector type of the
///   accessed \c simd_obj_impl derivative class object.
/// - If it is <code>overaligned_tag<N></code>, the location must be aligned by
///   \c N.
///
/// Program not meeting alignment requirements results in undefined
/// behavior.

/// @}

/// @addtogroup sycl_esimd_core_align
/// @{

/// \c element_aligned_tag type. Flag of this type should be used in load and
/// store operations when memory address is aligned by simd object's element
/// type.
struct element_aligned_tag {
  template <typename VT, typename ET = detail::element_type_t<VT>>
  static constexpr unsigned alignment = alignof(ET);
};

/// \c vector_aligned_tag type. Flag of this type should be used in load and
/// store operations when memory address is guaranteed to be aligned by simd
/// object's vector type.
struct vector_aligned_tag {
  template <typename VT> static constexpr unsigned alignment = alignof(VT);
};

/// \c overaligned_tag type. Flag of this type should be used in load and store
/// operations when memory address is aligned by the user-provided alignment
/// value N.
/// @tparam N is the alignment value. N must be a power of two.
template <unsigned N> struct overaligned_tag {
  static_assert(
      detail::isPowerOf2(N),
      "Alignment value N for overaligned_tag<N> must be a power of two");
  template <typename> static constexpr unsigned alignment = N;
};

inline constexpr element_aligned_tag element_aligned = {};

inline constexpr vector_aligned_tag vector_aligned = {};

template <unsigned N> inline constexpr overaligned_tag<N> overaligned = {};

/// Checks if type is a simd load/store flag.
template <typename T> struct is_simd_flag_type : std::false_type {};

template <> struct is_simd_flag_type<element_aligned_tag> : std::true_type {};

template <> struct is_simd_flag_type<vector_aligned_tag> : std::true_type {};

template <unsigned N>
struct is_simd_flag_type<overaligned_tag<N>> : std::true_type {};

/// Checks if given type is a simd load/store flag.
/// @tparam T the type to check
template <typename T>
static inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

/// @} sycl_esimd_core_align

namespace detail {

/// @cond ESIMD_DETAIL
/// \c dqword_element_aligned_tag type. Flag of this type should be used in load
/// and store operations when memory address is aligned by simd object's element
/// type or dword whatever is greater.
struct dqword_element_aligned_tag {
  template <typename VT, typename ET = detail::element_type_t<VT>>
  static constexpr unsigned alignment = alignof(ET) > 4 ? alignof(ET) : 4;
};

inline constexpr dqword_element_aligned_tag dqword_element_aligned = {};

// Functions to support efficient simd constructors - avoiding internal loop
// over elements.
template <class T, int N, size_t... Is>
constexpr vector_type_t<T, N> make_vector_impl(const T (&&Arr)[N],
                                               std::index_sequence<Is...>) {
  return vector_type_t<T, N>{Arr[Is]...};
}

template <class T, int N>
constexpr vector_type_t<T, N> make_vector(const T (&&Arr)[N]) {
  return make_vector_impl<T, N>(std::move(Arr), std::make_index_sequence<N>{});
}

template <class T, int N, size_t... Is>
constexpr auto make_vector_impl(T Base, T Stride, std::index_sequence<Is...>) {
  using CppT = typename element_type_traits<T>::EnclosingCppT;
  CppT BaseCpp = Base;
  CppT StrideCpp = Stride;
  vector_type_t<CppT, N> VBase = BaseCpp;
  vector_type_t<CppT, N> VStride = StrideCpp;
  vector_type_t<CppT, N> VStrideCoef{(CppT)(Is)...};
  vector_type_t<CppT, N> Result{VBase + VStride * VStrideCoef};
  return wrapper_type_converter<T>::template to_vector<N>(Result);
}

template <class T, int N> constexpr auto make_vector(T Base, T Stride) {
  return make_vector_impl<T, N>(Base, Stride, std::make_index_sequence<N>{});
}

/// @endcond ESIMD_DETAIL

/// @addtogroup sycl_esimd_core_vectors
/// @{

/// This is a base class for all ESIMD simd classes with real storage (simd,
/// simd_mask_impl). It wraps a clang vector as the storage for the elements.
/// Additionally this class supports region operations that map to Intel GPU
/// regions. The type of a region select or bit_cast_view operation is of
/// simd_view type, which models a "window" into this object's storage and can
/// used to read and modify it.
///
/// This class and its derivatives must follow the
/// 'curiously recurring template' design pattern.
///
/// @tparam RawTy Raw (storage) element type
/// @tparam N Number of elements
/// @tparam Derived - A class derived from this one. Pure \c simd_obj_impl
///   objects are never supposed to be constructed directly neither by user nor
///   by ESIMD library code, instead they should always be enclosed into objects
///   of some derived class - \c simd or \c simd_mask. This derived class is
///   captured by this template parameter.
///   Note that for some element types, the element type in the \c Derived
///   type and this type may differ - for example, \c half type.
/// @tparam SFINAE - defaults to 'void' in the forward declarion within
///   types.hpp, used to disable invalid specializations.
///
template <typename RawTy, int N, class Derived, class SFINAE>
#ifndef __SYCL_DEVICE_ONLY__
class simd_obj_impl {
#else
class [[__sycl_detail__::__uses_aspects__(
    sycl::aspect::ext_intel_esimd)]] simd_obj_impl {
#endif
  /// @cond ESIMD_DETAIL

  // For the is_simd_obj_impl_derivative helper to work correctly, all derived
  // classes must be templated by element type and number of elements. If fewer
  // template arguments are needed, template aliases can be used
  // (simd_mask_type).
  //
  template <typename, typename> friend class simd_view;
  template <typename, typename> friend class simd_view_impl;
  template <typename, int> friend class simd;
  template <typename, int> friend class simd_mask_impl;

  /// @endcond ESIMD_DETAIL

public:
  /// Element type of the derived (user) class.
  using element_type = get_vector_element_type<Derived>;

  /// The underlying raw storage vector data type.
  using raw_vector_type = vector_type_t<RawTy, N>;

  /// The element type of the raw storage vector.
  using raw_element_type = RawTy;

  /// The number of elements in this object.
  static constexpr int length = N;

protected:
  /// @cond ESIMD_DETAIL
  using Ty = element_type;

  template <bool UseSet = true>
  void init_from_array(const Ty (&&Arr)[N]) noexcept {
    raw_vector_type tmp;

    if constexpr (is_wrapper_elem_type_v<Ty>) {
      for (auto I = 0; I < N; ++I) {
        tmp[I] = bitcast_to_raw_type(Arr[I]);
      }
    } else {
      tmp = make_vector(std::move(Arr));
    }
    if constexpr (UseSet) {
      set(std::move(tmp));
    } else {
      M_data = std::move(tmp);
    }
  }

  explicit operator raw_vector_type() const {
    __esimd_dbg_print(explicit operator raw_vector_type());
    return data();
  }

private:
  Derived &cast_this_to_derived() { return reinterpret_cast<Derived &>(*this); }
  const Derived &cast_this_to_derived() const {
    return reinterpret_cast<const Derived &>(*this);
  }

  /// @endcond ESIMD_DETAIL

public:
  /// Default constructor. Values of the constructed object's elements are
  /// undefined.
  simd_obj_impl() = default;

  /// Copy constructor.
  /// @param other The other object to bitwise-copy elements from.
  simd_obj_impl(const simd_obj_impl &other) {
    __esimd_dbg_print(simd_obj_impl(const simd_obj_impl &other));
    set(other.data());
  }

  /// Implicit conversion constructor from another \c simd_obj_impl object.
  /// Elements of the of the other object are type-converted to \c element_type
  /// to obtain elements of this object.
  /// @tparam Ty1 Raw element type of the other object.
  /// @tparam Derived1 The actual type of the other object.
  /// @param other The other object.
  template <class Ty1, typename Derived1>
  simd_obj_impl(const simd_obj_impl<Ty1, N, Derived1, SFINAE> &other) {
    __esimd_dbg_print(simd_obj_impl(const simd_obj_impl... > &other));
    set(convert_vector<Ty, element_type_t<Derived1>, N>(other.data()));
  }

  /// Implicit conversion constructor from a raw vector object.
  /// @param Val the raw vector to convert from.
  simd_obj_impl(const raw_vector_type &Val) {
    __esimd_dbg_print(simd_obj_impl(const raw_vector_type &Val));
    set(Val);
  }

  /// Arithmetic progression constructor. Consecutive elements of this object
  /// are initialized with the arithmetic progression defined by the arguments.
  /// For example, <code>simd<int, 4> x(1, 3)</code> will initialize x to the
  /// <code>{1, 4, 7, 10}</code> sequence.
  /// If Ty is a floating-point type and \p Base or \p Step is +/-inf or nan,
  /// then this constructor has undefined behavior.
  /// @param Base The start of the progression.
  /// @param Step The step of the progression.
  simd_obj_impl(Ty Base, Ty Step) noexcept {
    __esimd_dbg_print(simd_obj_impl(Ty Base, Ty Step));
    M_data = make_vector<Ty, N>(Base, Step);
  }

  /// Broadcast constructor. Given value is type-converted to the
  /// \c element_type and resulting bit representation is broadcast to all lanes
  /// of the underlying vector.
  /// @tparam T1 Type of the value.
  /// @param Val The value to broadcast.
  template <class T1,
            class = std::enable_if_t<detail::is_valid_simd_elem_type_v<T1>>>
  simd_obj_impl(T1 Val) noexcept {
    __esimd_dbg_print(simd_obj_impl(T1 Val));
    M_data = bitcast_to_raw_type(detail::convert_scalar<Ty>(Val));
  }

  /// Rvalue array-based constructor. Used for in-place initialization like
  /// <code>simd<int, N> x({1,0,0,1,...})</code>.
  ///
  /// @param Arr Rvalue reference to an array of size @ref N to initialize from.
  template <int N1, class = std::enable_if_t<N1 == N>>
  simd_obj_impl(const Ty (&&Arr)[N1]) noexcept {
    __esimd_dbg_print(simd_obj_impl(const Ty(&&Arr)[N1]));
    init_from_array<false /*init M_data w/o using set(...)*/>(std::move(Arr));
    // It is OK not to mark a write to M_data with __esimd_vstore (via 'set')
    // here because:
    // - __esimd_vstore/vload are need only to mark ESIMD_PRIVATE variable
    //   access for the VC BE to generate proper code for them.
    // - initializers are not allowed for ESIMD_PRIVATE vars, so only the
    //   default ctor can be used for them
  }

  /// Pointer-based load constructor. Initializes this object from values stored
  /// in memory. For example:
  /// <code>simd<int, N> x(ptr, overaligned_tag<16>{});</code>.
  /// @tparam Flags Specifies memory address alignment. Affects efficiency of
  ///   the generated code.
  /// @param ptr The memory address to read from.
  template <typename Flags = element_aligned_tag,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  simd_obj_impl(const Ty *ptr, Flags = {}) noexcept {
    __esimd_dbg_print(simd_obj_impl(const Ty *ptr, Flags));
    copy_from(ptr, Flags{});
  }

  /// Accessor-based load constructor. Initializes constructed object from
  /// values stored in memory represented by an accessor and an offset. For
  /// example:
  /// <code>simd<int, N> x(acc, 128, overaligned_tag<16>{});</code>.
  /// @tparam AccessorT the type of the accessor. Auto-deduced.
  /// @tparam Flags Specifies memory address alignment. Affects efficiency of
  ///   the generated code. Auto-deduced from the unnamed alignment tag
  ///   argument.
  /// @param acc The accessor to read from.
  /// @param offset offset in bytes of the first element.
  template <
      typename AccessorT, typename Flags = element_aligned_tag,
      typename = std::enable_if_t<
          detail::is_accessor_with_v<AccessorT, accessor_mode_cap::can_read> &&
          is_simd_flag_type_v<Flags>>>
  simd_obj_impl(AccessorT acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
                uint64_t offset,
#else
                uint32_t offset,
#endif
                Flags = {}) noexcept {
    __esimd_dbg_print(simd_obj_impl(AccessorT acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
                                    uint64_t offset,
#else
                                    uint32_t offset,
#endif
                                    Flags));
    copy_from(acc, offset, Flags{});
  }

  /// Copy assignment operator.
  Derived &operator=(const simd_obj_impl &other) noexcept {
    set(other.data());
    return cast_this_to_derived();
  }

  /// Type conversion into a scalar:
  /// <code><simd_obj_impl<RawTy, 1, simd<Ty,1>></code> to \c Ty.
  template <typename T = simd_obj_impl,
            typename = std::enable_if_t<T::length == 1>>
  operator Ty() const {
    __esimd_dbg_print(operator Ty());
    return bitcast_to_wrapper_type<Ty>(data()[0]);
  }

  /// @return The value of the underlying raw vector.
  raw_vector_type data() const {
    __esimd_dbg_print(raw_vector_type data());
#ifndef __SYCL_DEVICE_ONLY__
    return M_data;
#else
    return __esimd_vload<RawTy, N>(&M_data);
#endif
  }

  /// @return A reference to the value of the
  /// underlying raw vector. Intended for use
  /// with l-value contexts in inline assembly.
  raw_vector_type &data_ref() { return M_data; }

  /// @return Newly constructed (from the underlying data) object of the Derived
  /// type.
  Derived read() const { return Derived{data()}; }

  /// Replaces the underlying data with the one taken from another object.
  /// @return This object.
  Derived &write(const Derived &Val) {
    set(Val.data());
    return cast_this_to_derived();
  }

  /// "Merges" this object's value with another object:
  /// replaces part of the underlying data with the one taken from the other
  /// object according to a mask. Only elements in lanes where corresponding
  /// mask's value is non-zero are replaced.
  /// @param Val The object to take new values from.
  /// @param Mask The mask.
  void merge(const Derived &Val, const simd_mask_type<N> &Mask) {
    set(__esimd_wrregion<RawTy, N, N, 0 /*VS*/, N, 1, N>(data(), Val.data(), 0,
                                                         Mask.data()));
  }

  /// Merges given two objects with a mask and writes resulting data into this
  /// object.
  /// @param Val1 The first object, provides elements for lanes with non-zero
  ///   corresponding predicates.
  /// @param Val2 The second object, provides elements for lanes with zero
  ///   corresponding predicates.
  /// @param Mask The merge mask.
  void merge(const Derived &Val1, Derived Val2, const simd_mask_type<N> &Mask) {
    Val2.merge(Val1, Mask);
    set(Val2.data());
  }

  /// Create a 1-dimensional view of this object.
  /// @tparam EltTy The element type of the new view.
  /// @return A new \c simd_view object which spans this entire object, but
  ///   potentially with a different element type and different number of
  ///   elements, if the sizes of this object's element type and the new one
  ///   don't match.
  template <typename EltTy> auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy = compute_format_type_t<Derived, EltTy>;
    using RetTy = simd_view<Derived, TopRegionTy>;
    return RetTy{cast_this_to_derived(), TopRegionTy{0}};
  }

  /// Create a 2-dimensional view of this object.
  /// <code>sizeof(EltTy)*Height*Width</code> must be equal to the byte size of
  /// this object.
  /// @tparam ElTy Element type of the view.
  /// @tparam Height Height of the view in rows.
  /// @tparam Width Width of the view in elements.
  /// @return A new 2D \c simd_view object which spans this entire object, but
  ///   potentially with a different element type and different number of
  ///   elements, if the sizes of this object's element type and the new one
  ///   don't match.
  template <typename EltTy, int Height, int Width>
  auto bit_cast_view() &[[clang::lifetimebound]] {
    using TopRegionTy = compute_format_type_2d_t<Derived, EltTy, Height, Width>;
    using RetTy = simd_view<Derived, TopRegionTy>;
    return RetTy{cast_this_to_derived(), TopRegionTy{0, 0}};
  }

  /// Select elements of this object into a subregion and create a 1D view for
  /// for it. Used when \c this is an lvalue.
  ///
  /// @tparam Size The number of elements selected for the subregion.
  /// @tparam Stride A distance in elements between two consecutive elements.
  /// @param Offset The starting element's offset.
  /// @return A view of the subregion.
  template <int Size, int Stride>
  simd_view<Derived, region1d_t<Ty, Size, Stride>>
  select(uint16_t Offset = 0) &[[clang::lifetimebound]] {
    static_assert(Size > 1 || Stride == 1,
                  "Stride must be 1 in single-element region");
    region1d_t<Ty, Size, Stride> Reg(Offset);
    return {cast_this_to_derived(), std::move(Reg)};
  }

  /// Select and extract a subregion of this object's elements and return it as
  /// a new vector object. Used when \c this is an rvalue.
  ///
  /// @tparam Size The number of elements selected for the subregion.
  /// @tparam Stride A distance in elements between two consecutive elements.
  /// @param Offset The starting element's offset.
  /// @return Extracted subregion as a new vector object.
  template <int Size, int Stride>
  resize_a_simd_type_t<Derived, Size> select(uint16_t Offset = 0) && {
    static_assert(Size > 1 || Stride == 1,
                  "Stride must be 1 in single-element region");
    Derived &&Val = std::move(cast_this_to_derived());
    return __esimd_rdregion<RawTy, N, Size, /*VS*/ 0, Size, Stride>(Val.data(),
                                                                    Offset);
  }

  /// Get value of this vector's element.
  /// @param i Element index.
  /// @return Value of i'th element.
  Ty operator[](int i) const { return bitcast_to_wrapper_type<Ty>(data()[i]); }

  /// Return writable view of a single element.
  /// @param i Element index.
  /// @return View of i'th element.
  simd_view<Derived, region1d_scalar_t<Ty>> operator[](int i)
      [[clang::lifetimebound]] {
    return select<1, 1>(i);
  }

  /// Indirect select - select and extract multiple elements with given
  /// variable indices.
  /// @tparam Size The number of elements to select.
  /// @param Indices Indices of element to select.
  /// @return Vector of extracted elements.
  template <int Size>
  resize_a_simd_type_t<Derived, Size>
  iselect(const simd<uint16_t, Size> &Indices) {
    vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(RawTy);
    return __esimd_rdindirect<RawTy, N, Size>(data(), Offsets);
  }

  /// Update single element with variable index.
  /// @param Index Element index.
  /// @param V New value.
  void iupdate(ushort Index, Ty V) {
    auto Val = data();
    Val[Index] = bitcast_to_raw_type(V);
    set(Val);
  }

  /// Indirect update - update multiple elements with given variable indices.
  /// @tparam Size The number of elements to update.
  /// @param Indices Indices of element to update.
  /// @param Val New values.
  /// @param Mask Operation mask. 1 - update, 0 - not.
  template <int Size>
  void iupdate(const simd<uint16_t, Size> &Indices,
               const resize_a_simd_type_t<Derived, Size> &Val,
               const simd_mask_type<Size> &Mask) {
    vector_type_t<uint16_t, Size> Offsets = Indices.data() * sizeof(RawTy);
    set(__esimd_wrindirect<RawTy, N, Size>(data(), Val.data(), Offsets,
                                           Mask.data()));
  }

  /// Replicates contents of this vector a number of times into a new vector.
  /// @tparam Rep The number of times this vector has to be replicated.
  /// @return Replicated simd_obj_impl instance.
  template <int Rep> resize_a_simd_type_t<Derived, Rep * N> replicate() const {
    return replicate_w<Rep, N>(0);
  }

  /// Shortcut to \c replicate_vs_w_hs with \c VS=0 and \c HS=1 to replicate a
  /// single "dense" (w/o gaps between elements) block \c Rep times.
  /// @tparam Rep The number of times to replicate the block.
  /// @tparam W Width - number of elements in the block.
  /// @param Offset Offset of the block's first element.
  /// @return Vector of size <code>Rep*W</code> consisting of replicated
  ///   elements of \c this object.
  template <int Rep, int W>
  resize_a_simd_type_t<Derived, Rep * W> replicate_w(uint16_t Offset) const {
    return replicate_vs_w_hs<Rep, 0, W, 1>(Offset);
  }

  /// Shortcut to \c replicate_vs_w_hs with \c HS=1 to replicate dense blocks.
  /// @tparam Rep Number of blocks to select for replication.
  /// @tparam VS Vertical stride - distance between first elements of
  ///   consecutive blocks. If \c VS=0, then the same block will be
  ///   replicated \c Rep times in the result.
  /// @tparam W Width - number of elements in a block.
  /// @param Offset The offset of the first element of the first block.
  /// @return Vector of size <code>Rep*W</code> consisting of replicated
  ///   elements of \c this object.
  template <int Rep, int VS, int W>
  resize_a_simd_type_t<Derived, Rep * W> replicate_vs_w(uint16_t Offset) const {
    return replicate_vs_w_hs<Rep, VS, W, 1>(Offset);
  }

  /// This function "replicates" a portion of this object's elements into a new
  /// object. The source elements to replicate are \c Rep number of blocks each
  /// of size \c W elements. Starting elements of consecutive blocks are \c VS
  /// elements apart and i'th block starts from
  /// <code>ith_block_start_ind = Offset + i*VS</code>
  /// index. Consecutive elements within a block are \c HS elements apart and
  /// j'th element in the block has <code>ith_block_start_ind + j*HS</code>
  /// index. Thus total of <code>Rep*W</code> elements are returned. Note that
  /// depending on \c VS, \c W and \c HS, blocks' elements may overlap and in
  /// this case the elements where the overlap happens may participate 2 or more
  /// times in the result.
  ///
  /// *Example 1*. Source object has 32 elements, \c Rep is 2, \c VS is 17, \c W
  /// is 3 and \c HS is 4. Selected elements are depicted with their index
  /// (matching their values) instead of a dot:
  /// @code
  /// simd<int, 32> Source(0/*Base*/, 1/*Step*/);
  /// simd<int, 6> Result = Source.replicate_vs_w_hs<2,17,3,4>(1);
  /// // |<-------------- VS=17 ------------->|
  /// //    v-------v-------v W=3
  /// //  . 1  . . . 5  . . . 9  . . . . . . . \ Rep=2
  /// //  . 18 . . . 22 . . . 26 . . . . .     /
  /// //   |<- HS=4->|
  /// // The Result is a vector of 6 source elements {1,5,9,18,22,26}.
  /// @endcode
  ///
  /// *Example 2*. AOS 7x3 => SOA 3x7 conversion.
  /// \c Rep is 3, \c VS is 1,  \c W is 7 and \c HS is 3.
  /// @code
  /// simd<float, 21> Source = getSource();
  /// simd<float, 21> Result = Source.replicate_vs_w_hs<3,1,7,3>(0);
  /// // Source:
  /// // x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6
  /// // Result:
  /// // x0 x1 x2 x3 x4 x5 x6 y0 y1 y2 y3 y4 y5 y6 z0 z1 z2 z3 z4 z5 z6
  /// @endcode
  ///
  /// @tparam Rep Number of blocks to select for replication.
  /// @tparam VS Vertical stride - distance between first elements of
  ///   consecutive blocks. If \c VS=0, then the same block will be
  ///   replicated \c Rep times in the result.
  /// @tparam W Width - number of elements in a block.
  /// @tparam HS Horizontal stride - distance between consecutive elements in a
  ///   block.
  /// @param Offset The offset of the first element of the first block.
  /// @return Vector of size <code>Rep*W</code> consisting of replicated
  ///   elements of \c this object.
  ///
  template <int Rep, int VS, int W, int HS>
  resize_a_simd_type_t<Derived, Rep * W>
  replicate_vs_w_hs(uint16_t Offset) const {
    return __esimd_rdregion<RawTy, N, Rep * W, VS, W, HS, N>(
        data(), Offset * sizeof(RawTy));
  }

  /// See if any element is non-zero.
  ///
  /// @return 1 if any element is non-zero, 0 otherwise.
  template <typename T1 = Ty,
            typename = std::enable_if_t<std::is_integral_v<T1>>>
  uint16_t any() const {
    return __esimd_any<Ty, N>(data());
  }

  /// See if all elements are non-zero.
  ///
  /// @return 1 if all elements are non-zero, 0 otherwise.
  template <typename T1 = Ty,
            typename = std::enable_if_t<std::is_integral_v<T1>>>
  uint16_t all() const {
    return __esimd_all<Ty, N>(data());
  }

protected:
  /// Write a simd_obj_impl-vector into a basic region of a simd_obj_impl
  /// object.
  template <typename RTy, class ElemTy = __raw_t<typename RTy::element_type>>
  ESIMD_INLINE void writeRegion(RTy Region,
                                const vector_type_t<ElemTy, RTy::length> &Val) {

    if constexpr (N * sizeof(RawTy) == RTy::length * sizeof(ElemTy))
      // update the entire vector
      set(bitcast<RawTy, ElemTy, RTy::length>(Val));
    else {
      static_assert(!RTy::Is_2D);
      // If element type differs, do bitcast conversion first.
      auto Base = bitcast<ElemTy, RawTy, N>(data());
      constexpr int BN = (N * sizeof(RawTy)) / sizeof(ElemTy);
      // Access the region information.
      constexpr int M = RTy::Size_x;
      constexpr int Stride = RTy::Stride_x;
      uint16_t Offset = Region.M_offset_x * sizeof(ElemTy);
      static_assert(M > 0, "Malformed RHS region.");
      static_assert(M <= BN, "Attempt to write beyond viewed area: The viewed "
                             "object in LHS does not fit RHS.");
      // (M > BN) condition is added below to not duplicate the above assert
      // for big values of M. The assert below is for 'Stride'.
      static_assert((M > BN) || (M - 1) * Stride < BN,
                    "Malformed RHS region - too big stride.");

      // Merge and update.
      auto Merged = __esimd_wrregion<ElemTy, BN, M,
                                     /*VS*/ 0, M, Stride>(Base, Val, Offset);
      // Convert back to the original element type, if needed.
      set(bitcast<RawTy, ElemTy, BN>(Merged));
    }
  }

  /// Write a simd_obj_impl-vector into a nested region of a simd_obj_impl
  /// object.
  template <typename TR, typename UR,
            class ElemTy = __raw_t<typename TR::element_type>>
  ESIMD_INLINE void writeRegion(std::pair<TR, UR> Region,
                                const vector_type_t<ElemTy, TR::length> &Val) {
    // parent-region type
    using PaTy = typename shape_type<UR>::type;
    using BT = __raw_t<typename PaTy::element_type>;
    constexpr int BN = PaTy::length;

    if constexpr (PaTy::Size_in_bytes == TR::Size_in_bytes) {
      writeRegion(Region.second, bitcast<BT, ElemTy, TR::length>(Val));
    } else {
      // Recursively read the base
      auto Base = readRegion<RawTy, N>(data(), Region.second);
      // If element type differs, do bitcast conversion first.
      auto Base1 = bitcast<ElemTy, BT, BN>(Base);
      constexpr int BN1 = PaTy::Size_in_bytes / sizeof(ElemTy);

      if constexpr (!TR::Is_2D) {
        // Access the region information.
        constexpr int M = TR::Size_x;
        constexpr int Stride = TR::Stride_x;
        uint16_t Offset = Region.first.M_offset_x * sizeof(ElemTy);

        static_assert(M <= BN1, "Attempt to write beyond viewed area: The "
                                "viewed object in LHS does not fit RHS.");
        static_assert(M > 0, "Malformed RHS region.");
        static_assert((M - 1) * Stride < BN,
                      "Malformed RHS region - too big stride.");
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

        static_assert(M <= BN1, "Attempt to write beyond viewed area: The "
                                "viewed object in LHS does not fit RHS.");
        static_assert(M > 0 && W > 0 && M % W == 0, "Malformed RHS region.");
        static_assert(W == 0 || ((M / W) - 1) * VS + (W - 1) * HS < BN1,
                      "Malformed RHS region - too big vertical and/or "
                      "horizontal stride.");
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

public:
  /// Copy a contiguous block of data from memory into this simd_obj_impl
  /// object. The amount of memory copied equals the total size of vector
  /// elements in this object.
  /// None of the template parameters except documented ones can/should be
  /// specified by callers.
  ///
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param addr the memory address to copy from. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  template <typename Flags = element_aligned_tag, int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE void copy_from(const Ty *addr, Flags = {}) SYCL_ESIMD_FUNCTION;

  /// Copy a contiguous block of data from memory into this simd_obj_impl
  /// object. The amount of memory copied equals the total size of vector
  /// elements in this object. Source memory location is represented via a
  /// global accessor and offset.
  /// None of the template parameters except documented ones can/should be
  /// specified by callers.
  /// @tparam AccessorT Type of the accessor (auto-deduced).
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from (in bytes).
  template <typename AccessorT, typename Flags = element_aligned_tag,
            int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_read, void>
  copy_from(AccessorT acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
            uint64_t offset,
#else
            uint32_t offset,
#endif
            Flags = {}) SYCL_ESIMD_FUNCTION;

  /// Copy a contiguous block of data from memory into this simd_obj_impl
  /// object. The amount of memory copied equals the total size of vector
  /// elements in this object. Source memory location is represented via a
  /// local accessor and offset.
  /// None of the template parameters except documented ones can/should be
  /// specified by callers.
  /// @tparam AccessorT Type of the accessor (auto-deduced).
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from (in bytes).
  template <typename AccessorT, typename Flags = element_aligned_tag,
            int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE std::enable_if_t<
      detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_read>,
      void>
  copy_from(AccessorT acc, uint32_t offset, Flags = {}) SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// None of the template parameters should be be specified by callers.
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param addr the memory address to copy to. Must be a pointer to the
  /// global address space, otherwise behavior is undefined.
  template <typename Flags = element_aligned_tag, int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE void copy_to(Ty *addr, Flags = {}) const SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// Destination memory location is represented via a global accessor and
  /// offset.
  /// None of the template parameters should be be specified by callers.
  /// @tparam AccessorT Type of the accessor (auto-deduced).
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from.
  template <typename AccessorT, typename Flags = element_aligned_tag,
            int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_write, void>
  copy_to(AccessorT acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
          uint64_t offset,
#else
          uint32_t offset,
#endif
          Flags = {}) const SYCL_ESIMD_FUNCTION;

  /// Copy all vector elements of this object into a contiguous block in memory.
  /// Destination memory location is represented via a local accessor and
  /// offset.
  /// None of the template parameters should be be specified by callers.
  /// @tparam AccessorT Type of the accessor (auto-deduced).
  /// @tparam Flags Alignment control for the copy operation.
  ///   See @ref sycl_esimd_core_align for more info.
  /// @param acc accessor to copy from.
  /// @param offset offset to copy from.
  template <typename AccessorT, typename Flags = element_aligned_tag,
            int ChunkSize = 32,
            typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
  ESIMD_INLINE std::enable_if_t<
      detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_write>,
      void>
  copy_to(AccessorT acc, uint32_t offset, Flags = {}) const SYCL_ESIMD_FUNCTION;

  // Unary operations.

  /// Per-element bitwise inversion, available in all subclasses, but only for
  /// integral element types (\c simd_mask included).
  /// @return Copy of this object with all elements bitwise inverted.
  template <class T1 = Ty, class = std::enable_if_t<std::is_integral_v<T1>>>
  Derived operator~() const {
    return Derived{
        detail::vector_unary_op<detail::UnaryOp::bit_not, T1, N>(data())};
  }

  /// Unary logical negation operator, available in all subclasses, but only for
  /// integral element types (\c simd_mask included).
  /// Similarly to C++, where !x returns bool, !simd returns a simd_mask, where
  /// each element is a result of comparision with zero.
  /// @return A \c simd_mask instance where each element is a result of
  ///   comparison of the original element with zero.
  template <class T1 = Ty, class = std::enable_if_t<std::is_integral_v<T1>>>
  simd_mask_type<N> operator!() const {
    return *this == 0;
  }

#define __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(BINOP, OPASSIGN, COND)              \
                                                                               \
  /** \c OPASSIGN @ref simd version.                                        */ \
  /** @tparam T1 Element type of the argument object (auto-deduced).        */ \
  /** @tparam SimdT The argument object type(auto-deduced).                 */ \
  /** @param RHS The argument object.                                       */ \
  template <class T1, class SimdT,                                             \
            class = std::enable_if_t<(is_simd_type_v<Derived> ==               \
                                      is_simd_type_v<SimdT>)&&COND>>           \
  Derived &operator OPASSIGN(                                                  \
      const __ESIMD_DNS::simd_obj_impl<T1, N, SimdT> &RHS) {                   \
    auto Res = *this BINOP RHS;                                                \
    using ResT = decltype(Res);                                                \
    set(__ESIMD_DNS::convert_vector<element_type, typename ResT::element_type, \
                                    length>(Res.data()));                      \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /** \c OPASSIGN @ref simd_view version.                                   */ \
  /** @tparam SimdT1 The type of the object "viewed" by the argument        */ \
  /**         (auto-deduced).                                               */ \
  /** @tparam RegionT1 Region type of the argument object (auto-deduced).   */ \
  /** @param RHS The argument object.                                       */ \
  template <class SimdT1, class RegionT1,                                      \
            class T1 = typename RegionT1::element_type,                        \
            class = std::enable_if_t<                                          \
                (is_simd_type_v<Derived> ==                                    \
                 is_simd_type_v<SimdT1>)&&(RegionT1::length == length) &&      \
                COND>>                                                         \
  Derived &operator OPASSIGN(                                                  \
      const __ESIMD_NS::simd_view<SimdT1, RegionT1> &RHS) {                    \
    auto Res = *this BINOP RHS.read();                                         \
    using ResT = decltype(Res);                                                \
    set(__ESIMD_DNS::convert_vector<element_type, typename ResT::element_type, \
                                    length>(Res.data()));                      \
    return cast_this_to_derived();                                             \
  }                                                                            \
                                                                               \
  /** \c OPASSIGN  scalar version.                                          */ \
  /** @tparam T1 The type of the scalar argument (auto-deduced).            */ \
  /** @param RHS The argument.                                              */ \
  template <class T1, class = std::enable_if_t<COND>>                          \
  Derived &operator OPASSIGN(T1 RHS) {                                         \
    if constexpr (is_simd_type_v<Derived>) {                                   \
      using RHSVecT = __ESIMD_DNS::construct_a_simd_type_t<Derived, T1, N>;    \
      return *this OPASSIGN RHSVecT(RHS);                                      \
    } else {                                                                   \
      return *this OPASSIGN Derived((RawTy)RHS);                               \
    }                                                                          \
  }

// Bitwise operations are defined for simd objects and masks, and both operands
// must be integral
#define __ESIMD_BITWISE_OP_FILTER                                              \
  std::is_integral_v<element_type> &&std::is_integral_v<T1>
  /// Bitwise-\c xor compound assignment. Available only when elements of both
  /// this object and the argument are integral.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(^, ^=, __ESIMD_BITWISE_OP_FILTER)
  /// Bitwise-\c or compound assignment. Available only when elements of both
  /// this object and the argument are integral.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(|, |=, __ESIMD_BITWISE_OP_FILTER)
  /// Bitwise-\c and compound assignment. Available only when elements of both
  /// this object and the argument are integral.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(&, &=, __ESIMD_BITWISE_OP_FILTER)
  /// Modulo operation compound assignment. Available only when elements of both
  /// this object and the argument are integral.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(%, %=, __ESIMD_BITWISE_OP_FILTER)
#undef __ESIMD_BITWISE_OP_FILTER

// Bit shift operations are defined only for simd objects (not for masks), and
// both operands must be integral
#define __ESIMD_SHIFT_OP_FILTER                                                \
  std::is_integral_v<element_type> &&std::is_integral_v<T1>                    \
      &&__ESIMD_DNS::is_simd_type_v<Derived>

  /// Shift left compound assignment. Available only when elements of both
  /// this object and the source are integral. Not available for \c simd_mask.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(<<, <<=, __ESIMD_SHIFT_OP_FILTER)
  /// Logical shift right compound assignment. Available only when elements of
  /// both this object and the source are integral. Not available for
  /// \c simd_mask.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(>>, >>=, __ESIMD_SHIFT_OP_FILTER)
#undef __ESIMD_SHIFT_OP_FILTER

// Arithmetic operations are defined only for simd objects, and the second
// operand's element type must be vectorizable. This requirement for 'this'
// is fulfilled, because otherwise 'this' couldn't have been constructed.
#define __ESIMD_ARITH_OP_FILTER                                                \
  __ESIMD_DNS::is_simd_type_v<Derived> &&__ESIMD_DNS::is_vectorizable_v<T1>

  /// Addition operation compound assignment.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(+, +=, __ESIMD_ARITH_OP_FILTER)
  /// Subtraction operation compound assignment.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(-, -=, __ESIMD_ARITH_OP_FILTER)
  /// Multiplication operation compound assignment.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(*, *=, __ESIMD_ARITH_OP_FILTER)
  /// Division operation compound assignment.
  __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN(/, /=, __ESIMD_ARITH_OP_FILTER)
#undef __ESIMD_ARITH_OP_FILTER
#undef __ESIMD_DEF_SIMD_OBJ_IMPL_OPASSIGN

  // Getter for the test proxy member, if enabled
  __ESIMD_DECLARE_TEST_PROXY_ACCESS

private:
  // The underlying data for this vector.
  raw_vector_type M_data;

  template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
  ESIMD_INLINE void copy_to_impl(AccessorT acc,
                                 TOffset offset) const SYCL_ESIMD_FUNCTION;
  template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
  ESIMD_INLINE void copy_from_impl(AccessorT acc,
                                   TOffset offset) SYCL_ESIMD_FUNCTION;

protected:
  // The test proxy if enabled
  __ESIMD_DECLARE_TEST_PROXY

  void set(const raw_vector_type &Val) {
#ifndef __SYCL_DEVICE_ONLY__
    M_data = Val;
#else
    __esimd_vstore<RawTy, N>(&M_data, Val);
#endif
  }
};
/// @} sycl_esimd_core_vectors

} // namespace detail

template <>
struct is_simd_flag_type<detail::dqword_element_aligned_tag> : std::true_type {
};
} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
