//==---------------- vector_core.hpp - sycl::vec class --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Check if Clang's ext_vector_type attribute is available. Host compiler
// may not be Clang, and Clang may not be built with the extension.
#ifdef __clang__
#ifndef __has_extension
#define __has_extension(x) 0
#endif
#ifndef __HAS_EXT_VECTOR_TYPE__
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif
#endif // __clang__

// See vec::DataType definitions for more details
#ifndef __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
#define __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE !__SYCL_USE_LIBSYCL8_VEC_IMPL
#endif

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/detail/vector_swizzle.hpp>

#include <sycl/detail/vector_base.hpp>

#include <sycl/detail/common.hpp>
#include <sycl/detail/fwd/accessor.hpp>
#include <sycl/detail/fwd/half.hpp>
#include <sycl/detail/memcpy.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec :
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
    public detail::vec_arith<DataT, NumElements>,
#else
    public detail::VecOperators<vec<DataT, NumElements>>::Combined,
#endif
    public detail::ApplyIf<
        NumElements == 1,
        detail::ScalarConversionOperatorsMixIn<vec<DataT, NumElements>>>,
    public detail::NamedSwizzlesMixinBoth<vec<DataT, NumElements>>,
    // Keep it last to simplify ABI layout test:
    public detail::vec_base<DataT, NumElements> {
  static_assert(std::is_same_v<DataT, std::remove_cv_t<DataT>>,
                "DataT must be cv-unqualified");

  static_assert(detail::is_allowed_vec_size_v<NumElements>,
                "Invalid number of elements for sycl::vec: only 1, 2, 3, 4, 8 "
                "or 16 are supported");
  static_assert(sizeof(bool) == sizeof(uint8_t), "bool size is not 1 byte");

  using Base = detail::vec_base<DataT, NumElements>;

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
#ifdef __SYCL_DEVICE_ONLY__
  using element_type_for_vector_t = typename detail::map_type<
      DataT,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
      std::byte, /*->*/ std::uint8_t, //
#endif
      bool, /*->*/ std::uint8_t,                            //
      sycl::half, /*->*/ sycl::detail::half_impl::StorageT, //
      sycl::ext::oneapi::bfloat16, /*->*/ uint16_t,         //
      char, /*->*/ detail::ConvertToOpenCLType_t<char>,     //
      DataT, /*->*/ DataT                                   //
      >::type;

public:
  // Type used for passing sycl::vec to SPIRV builtins.
  // We can not use ext_vector_type(1) as it's not supported by SPIRV
  // plugins (CTS fails).
  using vector_t =
      typename std::conditional_t<NumElements == 1, element_type_for_vector_t,
                                  element_type_for_vector_t __attribute__((
                                      ext_vector_type(NumElements)))>;

  // Make it a template to avoid ambiguity with `vec(const DataT &)` when
  // `vector_t` is the same as `DataT`. Not that the other ctor isn't a template
  // so we don't even need a smart `enable_if` condition here, the mere fact of
  // this being a template makes the other ctor preferred.
  // For vectors of length 3, make sure to only copy 3 elements, not 4, to work
  // around code generation issues, see LLVM #144454.
  template <
      typename vector_t_ = vector_t,
      typename = typename std::enable_if_t<std::is_same_v<vector_t_, vector_t>>>
  constexpr vec(vector_t_ openclVector) {
    sycl::detail::memcpy_no_adl(&this->m_Data, &openclVector,
                                NumElements *
                                    sizeof(element_type_for_vector_t));
  }

  /* @SYCL2020
   * Available only when: compiled for the device.
   * Converts this SYCL vec instance to the underlying backend-native vector
   * type defined by vector_t.
   */
  operator vector_t() const { return sycl::bit_cast<vector_t>(this->m_Data); }

private:
#endif // __SYCL_DEVICE_ONLY__
#endif

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
  template <int... Indexes>
  using Swizzle =
      detail::SwizzleOp<vec, detail::GetOp<DataT>, detail::GetOp<DataT>,
                        detail::GetOp, Indexes...>;

  template <int... Indexes>
  using ConstSwizzle =
      detail::SwizzleOp<const vec, detail::GetOp<DataT>, detail::GetOp<DataT>,
                        detail::GetOp, Indexes...>;
#else
  template <int... Indexes>
  using Swizzle =
      detail::hide_swizzle_from_adl::Swizzle<false, DataT, NumElements,
                                             Indexes...>;

  template <int... Indexes>
  using ConstSwizzle =
      detail::hide_swizzle_from_adl::Swizzle<true, DataT, NumElements,
                                             Indexes...>;
#endif

  // Element type for relational operator return value.
  using rel_t = detail::fixed_width_signed<sizeof(DataT)>;

public:
  // Aliases required by SYCL 2020 to make sycl::vec consistent
  // with that of marray and buffer.
  using element_type = DataT;
  using value_type = DataT;

  using Base::Base;
  constexpr vec(const vec &) = default;
  constexpr vec(vec &&) = default;

  /****************** Assignment Operators **************/
  constexpr vec &operator=(const vec &) = default;
  constexpr vec &operator=(vec &&) = default;

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
  // Template required to prevent ambiguous overload with the copy assignment
  // when NumElements == 1. The template prevents implicit conversion from
  // vec<_, 1> to DataT.
  template <typename Ty = DataT>
  typename std::enable_if_t<detail::is_fundamental_or_half_or_bfloat16<Ty>,
                            vec &>
  operator=(const DataT &Rhs) {
    *this = vec{Rhs};
    return *this;
  }

  // W/o this, things like "vec<char,*> = vec<signed char, *>" doesn't work.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      !std::is_same_v<Ty, rel_t> && std::is_convertible_v<Ty, rel_t>, vec &>
  operator=(const vec<rel_t, NumElements> &Rhs) {
    *this = Rhs.template as<vec>();
    return *this;
  }
#else
  template <typename T>
  typename std::enable_if_t<std::is_convertible_v<T, DataT>, vec &>
  operator=(const T &Rhs) {
    *this = vec{static_cast<DataT>(Rhs)};
    return *this;
  }
#endif

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(Base); }

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
private:
  // getValue should be able to operate on different underlying
  // types: enum cl_float#N , builtin vector float#N, builtin type float.
  constexpr auto getValue(int Index) const {
    using RetType =
        typename std::conditional_t<detail::is_byte_v<DataT>, int8_t,
#ifdef __SYCL_DEVICE_ONLY__
                                    element_type_for_vector_t
#else
                                    DataT
#endif
                                    >;

#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<DataT, sycl::ext::oneapi::bfloat16>)
      return sycl::bit_cast<RetType>(this->m_Data[Index]);
    else
#endif
      return static_cast<RetType>(this->m_Data[Index]);
  }

public:
#endif

  // Out-of-class definition is in `sycl/detail/vector_convert.hpp`
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const;

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
    return this;
#else
    return Swizzle<SwizzleIndexes...>{*this};
#endif
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
    return this;
#else
    return ConstSwizzle<SwizzleIndexes...>{*this};
#endif
  }

  const DataT &operator[](int i) const { return this->m_Data[i]; }

  DataT &operator[](int i) { return this->m_Data[i]; }

  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    for (int I = 0; I < NumElements; I++) {
      this->m_Data[I] = *multi_ptr<const DataT, Space, DecorateAddress>(
          Ptr + Offset * NumElements + I);
    }
  }
  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<DataT, Space, DecorateAddress> Ptr) {
    multi_ptr<const DataT, Space, DecorateAddress> ConstPtr(Ptr);
    load(Offset, ConstPtr);
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  load(size_t Offset,
       accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
           Acc) {
    multi_ptr<const DataT, detail::TargetToAS<Target>::AS,
              access::decorated::yes>
        MultiPtr(Acc);
    load(Offset, MultiPtr);
  }
  void load(size_t Offset, const DataT *Ptr) {
    for (int I = 0; I < NumElements; ++I)
      this->m_Data[I] = Ptr[Offset * NumElements + I];
  }

  template <access::address_space Space, access::decorated DecorateAddress>
  void store(size_t Offset,
             multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    for (int I = 0; I < NumElements; I++) {
      *multi_ptr<DataT, Space, DecorateAddress>(Ptr + Offset * NumElements +
                                                I) = this->m_Data[I];
    }
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  store(size_t Offset,
        accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
            Acc) {
    multi_ptr<DataT, detail::TargetToAS<Target>::AS, access::decorated::yes>
        MultiPtr(Acc);
    store(Offset, MultiPtr);
  }
  void store(size_t Offset, DataT *Ptr) const {
    for (int I = 0; I < NumElements; ++I)
      Ptr[Offset * NumElements + I] = this->m_Data[I];
  }

  // friends
  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class detail::SwizzleOp;
  template <typename T1, int T2> friend class __SYCL_EBO vec;
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
  // To allow arithmetic operators access private members of vec.
  template <typename T1, int T2> friend class detail::vec_arith;
#endif
};
///////////////////////// class sycl::vec /////////////////////////

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = std::enable_if_t<(std::is_same_v<T, U> && ...)>>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;
#endif

} // namespace _V1
} // namespace sycl