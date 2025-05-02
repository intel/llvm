//==---------------- vector.hpp --- Implements sycl::vec -------------------==//
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
#ifdef __HAS_EXT_VECTOR_TYPE__
#error "Undefine __HAS_EXT_VECTOR_TYPE__ macro"
#endif
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif // __clang__

// See vec::DataType definitions for more details
#ifndef __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
#define __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE !__SYCL_USE_LIBSYCL8_VEC_IMPL
#endif

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/memcpy.hpp>              // for memcpy
#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/type_traits.hpp> // for is_floating_point
#include <sycl/detail/vector_arith.hpp>
#include <sycl/half_type.hpp> // for StorageT, half, Vec16...

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#include <algorithm>   // for std::min
#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for size_t, NULL, byte
#include <cstdint>     // for uint8_t, int16_t, int...
#include <functional>  // for divides, multiplies
#include <iterator>    // for pair
#include <ostream>     // for operator<<, basic_ost...
#include <type_traits> // for enable_if_t, is_same
#include <utility>     // for index_sequence, make_...

namespace sycl {

// TODO: It should be within _V1 namespace, fix in the next ABI breaking
// windows.
enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {

struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

namespace detail {
// To be defined in tests, trick to access vec's private methods
template <typename T1, int T2> class vec_base_test;

template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp;

// Special type indicating that SwizzleOp should just read value from vector -
// not trying to perform any operations. Should not be called.
template <typename T> class GetOp {
public:
  using DataT = T;
  DataT getValue(size_t) const { return (DataT)0; }
  DataT operator()(DataT, DataT) { return (DataT)0; }
};

// Templated vs. non-templated conversion operator behaves differently when two
// conversions are needed as in the case below:
//
//   sycl::vec<int, 1> v;
//   std::ignore = static_cast<bool>(v);
//
// Make sure the snippet above compiles. That is important because
//
//   sycl::vec<int, 2> v;
//   if (v.x() == 42)
//     ...
//
// must go throw `v.x()` returning a swizzle, then its `operator==` returning
// vec<int, 1> and we want that code to compile.
template <typename Self> class ScalarConversionOperatorsMixIn {
  using element_type = typename from_incomplete<Self>::element_type;

public:
  operator element_type() const {
    return (*static_cast<const Self *>(this))[0];
  }

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
  template <
      typename T, typename = std::enable_if_t<!std::is_same_v<T, element_type>>,
      typename =
          std::void_t<decltype(static_cast<T>(std::declval<element_type>()))>>
  explicit operator T() const {
    return static_cast<T>((*static_cast<const Self *>(this))[0]);
  }
#endif
};

template <typename T>
inline constexpr bool is_fundamental_or_half_or_bfloat16 =
    std::is_fundamental_v<T> || std::is_same_v<std::remove_const_t<T>, half> ||
    std::is_same_v<std::remove_const_t<T>, ext::oneapi::bfloat16>;

// Per SYCL specification sycl::vec has different ctors available based on the
// number of elements. Without C++20's concepts we'd have to use partial
// specialization to represent that. This is a helper to do that. An alternative
// could be to have different specializations of the `sycl::vec` itself but then
// we'd need to outline all the common interfaces to re-use them.
template <typename DataT, int NumElements> class vec_base {
  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#memory-layout-and-alignment
  // It is required by the SPEC to align vec<DataT, 3> with vec<DataT, 4>.
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;
  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  //
  // In intel/llvm#14130 we incorrectly used std::array as an underlying storage
  // for vec data. The problem with std::array is that it comes from the C++
  // STL headers which we do not control and they may use something that is
  // illegal in SYCL device code. One of specific examples is use of debug
  // assertions in MSVC's STL implementation.
  //
  // The better approach is to use plain C++ array, but the problem here is that
  // C++ specification does not provide any guarantees about the memory layout
  // of std::array and therefore directly switching to it would technically be
  // an ABI-break, even though the practical chances of encountering the issue
  // are low.
  //
  // To play it safe, we only switch to use plain array if both its size and
  // alignment match those of std::array, or unless the new behavior is forced
  // via __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE or preview breaking changes mode.
  using DataType = std::conditional_t<
#if __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
      true,
#else
      sizeof(std::array<DataT, AdjustedNum>) == sizeof(DataT[AdjustedNum]) &&
          alignof(std::array<DataT, AdjustedNum>) ==
              alignof(DataT[AdjustedNum]),
#endif
      DataT[AdjustedNum], std::array<DataT, AdjustedNum>>;

  // To allow testing of private methods
  template <typename T1, int T2> friend class detail::vec_base_test;

protected:
  // fields
  // Alignment is the same as size, to a maximum size of 64. SPEC requires
  // "The elements of an instance of the SYCL vec class template are stored
  // in memory sequentially and contiguously and are aligned to the size of
  // the element type in bytes multiplied by the number of elements."
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

  template <size_t... Is>
  constexpr vec_base(const std::array<DataT, NumElements> &Arr,
                     std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

  template <typename CtorArgTy>
  static constexpr bool AllowArgTypeInVariadicCtor = []() constexpr {
    if constexpr (std::is_convertible_v<CtorArgTy, DataT>) {
      return true;
    } else if constexpr (is_vec_or_swizzle_v<CtorArgTy>) {
      if constexpr (CtorArgTy::size() == 1 &&
                    std::is_convertible_v<typename CtorArgTy::element_type,
                                          DataT>) {
        // Temporary workaround because swizzle's `operator DataT` is a
        // template.
        return true;
      }
      return std::is_same_v<typename CtorArgTy::element_type, DataT>;
    } else {
      return false;
    }
  }();

  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

  // Utility trait for creating an std::array from an vector argument.
  template <typename DataT_, typename T> class FlattenVecArg {
    template <std::size_t... Is>
    static constexpr auto helper(const T &V, std::index_sequence<Is...>) {
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
      // FIXME: Swizzle's `operator[]` for expression trees seems to be broken
      // and returns values of the underlying vector of some of the operands. On
      // the other hand, `getValue()` gives correct results. This can be changed
      // to using `operator[]` once the bug is fixed.
      if constexpr (is_swizzle_v<T>)
        return std::array{static_cast<DataT_>(V.getValue(Is))...};
      else
#endif
        return std::array{static_cast<DataT_>(V[Is])...};
    }

  public:
    constexpr auto operator()(const T &A) const {
      if constexpr (is_vec_or_swizzle_v<T>) {
        return helper(A, std::make_index_sequence<T ::size()>());
      } else {
        return std::array{static_cast<DataT_>(A)};
      }
    }
  };

  // Alias for shortening the vec arguments to array converter.
  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator = ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

public:
  constexpr vec_base() = default;
  constexpr vec_base(const vec_base &) = default;
  constexpr vec_base(vec_base &&) = default;
  constexpr vec_base &operator=(const vec_base &) = default;
  constexpr vec_base &operator=(vec_base &&) = default;

  explicit constexpr vec_base(const DataT &arg)
      : vec_base(RepeatValue<NumElements>(arg),
                 std::make_index_sequence<NumElements>()) {}

  template <typename... argTN,
            typename = std::enable_if_t<
                ((AllowArgTypeInVariadicCtor<argTN> && ...)) &&
                ((num_elements<argTN>() + ...)) == NumElements>>
  constexpr vec_base(const argTN &...args)
      : vec_base{VecArgArrayCreator<DataT, argTN...>::Create(args...),
                 std::make_index_sequence<NumElements>()} {}
};

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
template <typename DataT> class vec_base<DataT, 1> {
  using DataType = std::conditional_t<
#if __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
      true,
#else
      sizeof(std::array<DataT, 1>) == sizeof(DataT[1]) &&
          alignof(std::array<DataT, 1>) == alignof(DataT[1]),
#endif
      DataT[1], std::array<DataT, 1>>;

protected:
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

public:
  constexpr vec_base() = default;
  constexpr vec_base(const vec_base &) = default;
  constexpr vec_base(vec_base &&) = default;
  constexpr vec_base &operator=(const vec_base &) = default;
  constexpr vec_base &operator=(vec_base &&) = default;

  // Not `explicit` on purpose, differs from NumElements > 1.
  constexpr vec_base(const DataT &arg) : m_Data{{arg}} {}
};

template <typename Self> class ConversionToVecMixin {
  using vec_ty = typename from_incomplete<Self>::result_vec_ty;

public:
  operator vec_ty() const {
    auto &self = *static_cast<const Self *>(this);
    if constexpr (vec_ty::size() == 1)
      // Avoid recursion by explicitly going through `vec(const DataT &)` ctor.
      return vec_ty{static_cast<typename vec_ty::element_type>(self)};
    else
      // Uses `vec`'s variadic ctor.
      return vec_ty{self};
  }
};

template <typename Self, typename = void> class SwizzleBase {
  using VecT = typename from_incomplete<Self>::vec_ty;

public:
  explicit SwizzleBase(VecT &Vec) : Vec(Vec) {}

  const Self &operator=(const Self &) = delete;

protected:
  VecT &Vec;
};

template <typename Self>
class SwizzleBase<Self,
                  std::enable_if_t<from_incomplete<Self>::is_assignable>> {
  using VecT = typename from_incomplete<Self>::vec_ty;
  using ResultVecT = typename from_incomplete<Self>::result_vec_ty;

  using DataT = typename from_incomplete<Self>::element_type;
  static constexpr int N = from_incomplete<Self>::size();

public:
  explicit SwizzleBase(VecT &Vec) : Vec(Vec) {}

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) const {
    ResultVecT v;
    v.load(offset, ptr);
    *static_cast<Self *>(this) = v;
  }

  template <bool OtherIsConstVec, int OtherVecSize, int... OtherIndexes>
  std::enable_if_t<sizeof...(OtherIndexes) == N, const Self &>
  operator=(const detail::hide_swizzle_from_adl::Swizzle<
            OtherIsConstVec, DataT, OtherVecSize, OtherIndexes...> &rhs) {
    return (*this = static_cast<ResultVecT>(rhs));
  }

  const Self &operator=(const ResultVecT &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs[i];

    return *static_cast<const Self *>(this);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, DataT> &&
                                        !is_swizzle_v<T>>>
  const Self &operator=(const T &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = static_cast<DataT>(rhs);

    return *static_cast<const Self *>(this);
  }

  // Default copy-assignment. Self's implicitly generated copy-assignment uses
  // this.
  //
  // We're templated on "Self", so each swizzle has its own SwizzleBase and the
  // following is ok (1-to-1 bidirectional mapping between Self and its
  // SwizzleBase instantiation) even if a bit counterintuitive.
  const SwizzleBase &operator=(const SwizzleBase &rhs) const {
    const Self &self = (*static_cast<const Self *>(this));
    self = static_cast<ResultVecT>(static_cast<const Self &>(rhs));
    return self;
  }

protected:
  VecT &Vec;
};

namespace hide_swizzle_from_adl {
// Can't have sycl::vec anywhere in template parameters because that would bring
// its hidden friends into ADL. Put it in a dedicated namespace to avoid
// anything extra via ADL as well.
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
class __SYCL_EBO Swizzle
    : public SwizzleBase<Swizzle<IsConstVec, DataT, VecSize, Indexes...>>,
      public SwizzleOperators<
          Swizzle<IsConstVec, DataT, VecSize, Indexes...>>::Combined,
      public ApplyIf<sizeof...(Indexes) == 1,
                     ScalarConversionOperatorsMixIn<
                         Swizzle<IsConstVec, DataT, VecSize, Indexes...>>>,
      public ConversionToVecMixin<
          Swizzle<IsConstVec, DataT, VecSize, Indexes...>>,
      public NamedSwizzlesMixinBoth<
          Swizzle<IsConstVec, DataT, VecSize, Indexes...>> {
  using Base = SwizzleBase<Swizzle<IsConstVec, DataT, VecSize, Indexes...>>;

  static constexpr int NumElements = sizeof...(Indexes);
  using ResultVec = vec<DataT, NumElements>;

  // Get underlying vec index for (*this)[idx] access.
  static constexpr auto get_vec_idx(int idx) {
    int counter = 0;
    int result = -1;
    ((result = counter++ == idx ? Indexes : result), ...);
    return result;
  }

public:
  using Base::Base;
  using Base::operator=;

  using element_type = DataT;
  using value_type = DataT;

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = typename vec<DataT, NumElements>::vector_t;
#endif // __SYCL_DEVICE_ONLY__
#endif

  Swizzle() = delete;
  Swizzle(const Swizzle &) = delete;

  static constexpr size_t byte_size() noexcept {
    return ResultVec::byte_size();
  }
  static constexpr size_t size() noexcept { return ResultVec::size(); }

  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const { return static_cast<ResultVec>(*this).get_size(); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const {
    return static_cast<ResultVec>(*this).get_count();
  };

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    return static_cast<ResultVec>(*this)
        .template convert<ConvertT, RoundingMode>();
  }

  template <typename asT> asT as() const {
    return static_cast<ResultVec>(*this).template as<asT>();
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
    return static_cast<ResultVec>(*this).store(offset, ptr);
  }

  template <int... swizzleIndexes> auto swizzle() const {
    return this->Vec.template swizzle<get_vec_idx(swizzleIndexes)...>();
  }

  auto &operator[](int index) const { return this->Vec[get_vec_idx(index)]; }
};
} // namespace hide_swizzle_from_adl
#endif
} // namespace detail

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
      sycl::ext::oneapi::bfloat16,
      /*->*/ sycl::ext::oneapi::bfloat16::Bfloat16StorageT, //
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
  template <
      typename vector_t_ = vector_t,
      typename = typename std::enable_if_t<std::is_same_v<vector_t_, vector_t>>>
  constexpr vec(vector_t_ openclVector) {
    sycl::detail::memcpy_no_adl(&this->m_Data, &openclVector,
                                sizeof(openclVector));
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

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
namespace detail {

// Special type for working SwizzleOp with scalars, stores a scalar and gives
// the scalar at any index. Provides interface is compatible with SwizzleOp
// operations
template <typename T> class GetScalarOp {
public:
  using DataT = T;
  GetScalarOp(DataT Data) : m_Data(Data) {}
  DataT getValue(size_t) const { return m_Data; }

private:
  DataT m_Data;
};
template <typename T> using rel_t = detail::fixed_width_signed<sizeof(T)>;

template <typename T> struct EqualTo {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs == Rhs) ? -1 : 0;
  }
};

template <typename T> struct NotEqualTo {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs != Rhs) ? -1 : 0;
  }
};

template <typename T> struct GreaterEqualTo {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs >= Rhs) ? -1 : 0;
  }
};

template <typename T> struct LessEqualTo {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs <= Rhs) ? -1 : 0;
  }
};

template <typename T> struct GreaterThan {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs > Rhs) ? -1 : 0;
  }
};

template <typename T> struct LessThan {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs < Rhs) ? -1 : 0;
  }
};

template <typename T> struct LogicalAnd {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs && Rhs) ? -1 : 0;
  }
};

template <typename T> struct LogicalOr {
  constexpr rel_t<T> operator()(const T &Lhs, const T &Rhs) const {
    return (Lhs || Rhs) ? -1 : 0;
  }
};

template <typename T> struct RShift {
  constexpr T operator()(const T &Lhs, const T &Rhs) const {
    return Lhs >> Rhs;
  }
};

template <typename T> struct LShift {
  constexpr T operator()(const T &Lhs, const T &Rhs) const {
    return Lhs << Rhs;
  }
};

///////////////////////// class SwizzleOp /////////////////////////
// SwizzleOP represents expression templates that operate on vec.
// Actual computation performed on conversion or assignment operators.
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp : public detail::NamedSwizzlesMixinBoth<
                      SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                OperationCurrentT, Indexes...>,
                      sizeof...(Indexes)> {
  using DataT = typename VecT::element_type;

public:
  using element_type = DataT;
  using value_type = DataT;

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  static constexpr size_t size() noexcept { return sizeof...(Indexes); }

  template <int Num = size()>
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const {
    return byte_size<Num>();
  }

  template <int Num = size()> size_t byte_size() const noexcept {
    return sizeof(DataT) * (Num == 3 ? 4 : Num);
  }

private:
  // Certain operators return a vector with a different element type. Also, the
  // left and right operand types may differ. CommonDataT selects a result type
  // based on these types to ensure that the result value can be represented.
  //
  // Example 1:
  //   sycl::vec<unsigned char, 4> vec{...};
  //   auto result = 300u + vec.x();
  //
  // CommonDataT is std::common_type_t<OperationLeftT, OperationRightT> since
  // it's larger than unsigned char.
  //
  // Example 2:
  //   sycl::vec<bool, 1> vec{...};
  //   auto result = vec.template swizzle<sycl::elem::s0>() && vec;
  //
  // CommonDataT is DataT since operator&& returns a vector with element type
  // int8_t, which is larger than bool.
  //
  // Example 3:
  //   sycl::vec<std::byte, 4> vec{...}; auto swlo = vec.lo();
  //   auto result = swlo == swlo;
  //
  // CommonDataT is DataT since operator== returns a vector with element type
  // int8_t, which is the same size as std::byte. std::common_type_t<DataT, ...>
  // can't be used here since there's no type that int8_t and std::byte can both
  // be implicitly converted to.
  using OpLeftDataT = typename OperationLeftT::DataT;
  using OpRightDataT = typename OperationRightT::DataT;
  using CommonDataT = std::conditional_t<
      sizeof(DataT) >= sizeof(std::common_type_t<OpLeftDataT, OpRightDataT>),
      DataT, std::common_type_t<OpLeftDataT, OpRightDataT>>;

  using rel_t = detail::rel_t<DataT>;
  using vec_t = vec<DataT, sizeof...(Indexes)>;
  using vec_rel_t = vec<rel_t, sizeof...(Indexes)>;

  template <typename OperationRightT_,
            template <typename> class OperationCurrentT_, int... Idx_>
  using NewLHOp = SwizzleOp<VecT,
                            SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                      OperationCurrentT, Indexes...>,
                            OperationRightT_, OperationCurrentT_, Idx_...>;

  template <typename OperationRightT_,
            template <typename> class OperationCurrentT_, int... Idx_>
  using NewRelOp = SwizzleOp<vec<rel_t, VecT::size()>,
                             SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                       OperationCurrentT, Indexes...>,
                             OperationRightT_, OperationCurrentT_, Idx_...>;

  template <typename OperationLeftT_,
            template <typename> class OperationCurrentT_, int... Idx_>
  using NewRHOp = SwizzleOp<VecT, OperationLeftT_,
                            SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                      OperationCurrentT, Indexes...>,
                            OperationCurrentT_, Idx_...>;

  template <int IdxNum, typename T = void>
  using EnableIfOneIndex =
      typename std::enable_if_t<1 == IdxNum && SwizzleOp::size() == IdxNum, T>;

  template <int IdxNum, typename T = void>
  using EnableIfMultipleIndexes =
      typename std::enable_if_t<1 != IdxNum && SwizzleOp::size() == IdxNum, T>;

  template <typename T>
  using EnableIfScalarType =
      typename std::enable_if_t<std::is_convertible_v<DataT, T> &&
                                detail::is_fundamental_or_half_or_bfloat16<T>>;

  template <typename T>
  using EnableIfNoScalarType =
      typename std::enable_if_t<!std::is_convertible_v<DataT, T> ||
                                !detail::is_fundamental_or_half_or_bfloat16<T>>;

  template <int... Indices>
  using Swizzle =
      SwizzleOp<VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

  template <int... Indices>
  using ConstSwizzle =
      SwizzleOp<const VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

public:
#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = typename vec_t::vector_t;
#endif // __SYCL_DEVICE_ONLY__

  const DataT &operator[](int i) const {
    std::array<int, size()> Idxs{Indexes...};
    return (*m_Vector)[Idxs[i]];
  }

  template <typename _T = VecT>
  std::enable_if_t<!std::is_const_v<_T>, DataT> &operator[](int i) {
    std::array<int, size()> Idxs{Indexes...};
    return (*m_Vector)[Idxs[i]];
  }

  template <typename T, int IdxNum = size(),
            typename = EnableIfOneIndex<IdxNum>,
            typename = EnableIfScalarType<T>>
  operator T() const {
    return getValue(0);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  friend NewRHOp<GetScalarOp<T>, std::multiplies, Indexes...>
  operator*(const T &Lhs, const SwizzleOp &Rhs) {
    return NewRHOp<GetScalarOp<T>, std::multiplies, Indexes...>(
        Rhs.m_Vector, GetScalarOp<T>(Lhs), Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  friend NewRHOp<GetScalarOp<T>, std::plus, Indexes...>
  operator+(const T &Lhs, const SwizzleOp &Rhs) {
    return NewRHOp<GetScalarOp<T>, std::plus, Indexes...>(
        Rhs.m_Vector, GetScalarOp<T>(Lhs), Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  friend NewRHOp<GetScalarOp<T>, std::divides, Indexes...>
  operator/(const T &Lhs, const SwizzleOp &Rhs) {
    return NewRHOp<GetScalarOp<T>, std::divides, Indexes...>(
        Rhs.m_Vector, GetScalarOp<T>(Lhs), Rhs);
  }

  // TODO: Check that Rhs arg is suitable.
#ifdef __SYCL_OPASSIGN
#error "Undefine __SYCL_OPASSIGN macro."
#endif
#define __SYCL_OPASSIGN(OPASSIGN, OP)                                          \
  friend const SwizzleOp &operator OPASSIGN(const SwizzleOp & Lhs,             \
                                            const DataT & Rhs) {               \
    Lhs.operatorHelper<OP>(vec_t(Rhs));                                        \
    return Lhs;                                                                \
  }                                                                            \
  template <typename RhsOperation>                                             \
  friend const SwizzleOp &operator OPASSIGN(const SwizzleOp & Lhs,             \
                                            const RhsOperation & Rhs) {        \
    Lhs.operatorHelper<OP>(Rhs);                                               \
    return Lhs;                                                                \
  }                                                                            \
  friend const SwizzleOp &operator OPASSIGN(const SwizzleOp & Lhs,             \
                                            const vec_t & Rhs) {               \
    Lhs.operatorHelper<OP>(Rhs);                                               \
    return Lhs;                                                                \
  }

  __SYCL_OPASSIGN(+=, std::plus)
  __SYCL_OPASSIGN(-=, std::minus)
  __SYCL_OPASSIGN(*=, std::multiplies)
  __SYCL_OPASSIGN(/=, std::divides)
  __SYCL_OPASSIGN(%=, std::modulus)
  __SYCL_OPASSIGN(&=, std::bit_and)
  __SYCL_OPASSIGN(|=, std::bit_or)
  __SYCL_OPASSIGN(^=, std::bit_xor)
  __SYCL_OPASSIGN(>>=, RShift)
  __SYCL_OPASSIGN(<<=, LShift)
#undef __SYCL_OPASSIGN

#ifdef __SYCL_UOP
#error "Undefine __SYCL_UOP macro"
#endif
#define __SYCL_UOP(UOP, OPASSIGN)                                              \
  friend const SwizzleOp &operator UOP(const SwizzleOp & sv) {                 \
    sv OPASSIGN static_cast<DataT>(1);                                         \
    return sv;                                                                 \
  }                                                                            \
  friend vec_t operator UOP(const SwizzleOp &sv, int) {                        \
    vec_t Ret = sv;                                                            \
    sv OPASSIGN static_cast<DataT>(1);                                         \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  template <typename T = DataT>
  friend typename std::enable_if_t<
      std::is_same_v<T, DataT> && !detail::is_vgenfloat_v<T>, vec_t>
  operator~(const SwizzleOp &Rhs) {
    vec_t Tmp = Rhs;
    return ~Tmp;
  }

  friend vec_rel_t operator!(const SwizzleOp &Rhs) {
    vec_t Tmp = Rhs;
    return !Tmp;
  }

  friend vec_t operator+(const SwizzleOp &Rhs) {
    vec_t Tmp = Rhs;
    return +Tmp;
  }

  friend vec_t operator-(const SwizzleOp &Rhs) {
    vec_t Tmp = Rhs;
    return -Tmp;
  }

// scalar BINOP vec<>
// scalar BINOP SwizzleOp
// vec<> BINOP SwizzleOp
#ifdef __SYCL_BINOP
#error "Undefine __SYCL_BINOP macro"
#endif
#define __SYCL_BINOP(BINOP, COND)                                              \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(                       \
      const DataT & Lhs, const SwizzleOp & Rhs) {                              \
    vec_t Tmp = Rhs;                                                           \
    return Lhs BINOP Tmp;                                                      \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const SwizzleOp & Lhs, \
                                                        const DataT & Rhs) {   \
    vec_t Tmp = Lhs;                                                           \
    return Tmp BINOP Rhs;                                                      \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(                       \
      const vec_t & Lhs, const SwizzleOp & Rhs) {                              \
    vec_t Tmp = Rhs;                                                           \
    return Lhs BINOP Tmp;                                                      \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const SwizzleOp & Lhs, \
                                                        const vec_t & Rhs) {   \
    vec_t Tmp = Lhs;                                                           \
    return Tmp BINOP Rhs;                                                      \
  }

  __SYCL_BINOP(+, (!detail::is_byte_v<T>))
  __SYCL_BINOP(-, (!detail::is_byte_v<T>))
  __SYCL_BINOP(*, (!detail::is_byte_v<T>))
  __SYCL_BINOP(/, (!detail::is_byte_v<T>))
  __SYCL_BINOP(%, (!detail::is_byte_v<T>))
  __SYCL_BINOP(&, true)
  __SYCL_BINOP(|, true)
  __SYCL_BINOP(^, true)
  // We have special <<, >> operators for std::byte.
  __SYCL_BINOP(>>, (!detail::is_byte_v<T>))
  __SYCL_BINOP(<<, (!detail::is_byte_v<T>))

  template <typename T = DataT>
  friend std::enable_if_t<detail::is_byte_v<T>, vec_t>
  operator>>(const SwizzleOp &Lhs, const int shift) {
    vec_t Tmp = Lhs;
    return Tmp >> shift;
  }

  template <typename T = DataT>
  friend std::enable_if_t<detail::is_byte_v<T>, vec_t>
  operator<<(const SwizzleOp &Lhs, const int shift) {
    vec_t Tmp = Lhs;
    return Tmp << shift;
  }
#undef __SYCL_BINOP

// scalar RELLOGOP vec<>
// scalar RELLOGOP SwizzleOp
// vec<> RELLOGOP SwizzleOp
#ifdef __SYCL_RELLOGOP
#error "Undefine __SYCL_RELLOGOP macro"
#endif
#define __SYCL_RELLOGOP(RELLOGOP, COND)                                        \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_rel_t> operator RELLOGOP(                \
      const DataT & Lhs, const SwizzleOp & Rhs) {                              \
    vec_t Tmp = Rhs;                                                           \
    return Lhs RELLOGOP Tmp;                                                   \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_rel_t> operator RELLOGOP(                \
      const SwizzleOp & Lhs, const DataT & Rhs) {                              \
    vec_t Tmp = Lhs;                                                           \
    return Tmp RELLOGOP Rhs;                                                   \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_rel_t> operator RELLOGOP(                \
      const vec_t & Lhs, const SwizzleOp & Rhs) {                              \
    vec_t Tmp = Rhs;                                                           \
    return Lhs RELLOGOP Tmp;                                                   \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_rel_t> operator RELLOGOP(                \
      const SwizzleOp & Lhs, const vec_t & Rhs) {                              \
    vec_t Tmp = Lhs;                                                           \
    return Tmp RELLOGOP Rhs;                                                   \
  }

  __SYCL_RELLOGOP(==, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(!=, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(>, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(<, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(>=, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(<=, (!detail::is_byte_v<T>))
  __SYCL_RELLOGOP(&&, (!detail::is_byte_v<T> && !detail::is_vgenfloat_v<T>))
  __SYCL_RELLOGOP(||, (!detail::is_byte_v<T> && !detail::is_vgenfloat_v<T>))
#undef __SYCL_RELLOGOP

  template <int IdxNum = size(), typename = EnableIfMultipleIndexes<IdxNum>>
  SwizzleOp &operator=(const vec<DataT, IdxNum> &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      (*m_Vector)[Idxs[I]] = Rhs[I];
    }
    return *this;
  }

  template <int IdxNum = size(), typename = EnableIfOneIndex<IdxNum>>
  SwizzleOp &operator=(const DataT &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    (*m_Vector)[Idxs[0]] = Rhs;
    return *this;
  }

  template <int IdxNum = size(), EnableIfMultipleIndexes<IdxNum, bool> = true>
  SwizzleOp &operator=(const DataT &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    for (auto Idx : Idxs) {
      (*m_Vector)[Idx] = Rhs;
    }
    return *this;
  }

  template <int IdxNum = size(), typename = EnableIfOneIndex<IdxNum>>
  SwizzleOp &operator=(DataT &&Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    (*m_Vector)[Idxs[0]] = Rhs;
    return *this;
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::multiplies, Indexes...>
  operator*(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::multiplies, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::multiplies, Indexes...>
  operator*(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::multiplies, Indexes...>(m_Vector, *this,
                                                              Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::plus, Indexes...> operator+(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::plus, Indexes...>(m_Vector, *this,
                                                          GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::plus, Indexes...>
  operator+(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::plus, Indexes...>(m_Vector, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::minus, Indexes...>
  operator-(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::minus, Indexes...>(m_Vector, *this,
                                                           GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::minus, Indexes...>
  operator-(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::minus, Indexes...>(m_Vector, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::divides, Indexes...>
  operator/(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::divides, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::divides, Indexes...>
  operator/(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::divides, Indexes...>(m_Vector, *this,
                                                           Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::modulus, Indexes...>
  operator%(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::modulus, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::modulus, Indexes...>
  operator%(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::modulus, Indexes...>(m_Vector, *this,
                                                           Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::bit_and, Indexes...>
  operator&(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::bit_and, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::bit_and, Indexes...>
  operator&(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::bit_and, Indexes...>(m_Vector, *this,
                                                           Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::bit_or, Indexes...>
  operator|(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::bit_or, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::bit_or, Indexes...>
  operator|(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::bit_or, Indexes...>(m_Vector, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, std::bit_xor, Indexes...>
  operator^(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, std::bit_xor, Indexes...>(
        m_Vector, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, std::bit_xor, Indexes...>
  operator^(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, std::bit_xor, Indexes...>(m_Vector, *this,
                                                           Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, RShift, Indexes...> operator>>(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, RShift, Indexes...>(m_Vector, *this,
                                                       GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, RShift, Indexes...>
  operator>>(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, RShift, Indexes...>(m_Vector, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewLHOp<GetScalarOp<T>, LShift, Indexes...> operator<<(const T &Rhs) const {
    return NewLHOp<GetScalarOp<T>, LShift, Indexes...>(m_Vector, *this,
                                                       GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewLHOp<RhsOperation, LShift, Indexes...>
  operator<<(const RhsOperation &Rhs) const {
    return NewLHOp<RhsOperation, LShift, Indexes...>(m_Vector, *this, Rhs);
  }

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5,
            typename = typename std::enable_if_t<sizeof...(T5) == size()>>
  SwizzleOp &operator=(const SwizzleOp<T1, T2, T3, T4, T5...> &Rhs) {
    std::array<int, size()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      (*m_Vector)[Idxs[I]] = Rhs.getValue(I);
    }
    return *this;
  }

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5,
            typename = typename std::enable_if_t<sizeof...(T5) == size()>>
  SwizzleOp &operator=(SwizzleOp<T1, T2, T3, T4, T5...> &&Rhs) {
    std::array<int, size()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      (*m_Vector)[Idxs[I]] = Rhs.getValue(I);
    }
    return *this;
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, EqualTo, Indexes...> operator==(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, EqualTo, Indexes...>(NULL, *this,
                                                         GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, EqualTo, Indexes...>
  operator==(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, EqualTo, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, NotEqualTo, Indexes...>
  operator!=(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, NotEqualTo, Indexes...>(
        NULL, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, NotEqualTo, Indexes...>
  operator!=(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, NotEqualTo, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, GreaterEqualTo, Indexes...>
  operator>=(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, GreaterEqualTo, Indexes...>(
        NULL, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, GreaterEqualTo, Indexes...>
  operator>=(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, GreaterEqualTo, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, LessEqualTo, Indexes...>
  operator<=(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, LessEqualTo, Indexes...>(
        NULL, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, LessEqualTo, Indexes...>
  operator<=(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, LessEqualTo, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, GreaterThan, Indexes...>
  operator>(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, GreaterThan, Indexes...>(
        NULL, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, GreaterThan, Indexes...>
  operator>(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, GreaterThan, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, LessThan, Indexes...> operator<(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, LessThan, Indexes...>(NULL, *this,
                                                          GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, LessThan, Indexes...>
  operator<(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, LessThan, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, LogicalAnd, Indexes...>
  operator&&(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, LogicalAnd, Indexes...>(
        NULL, *this, GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, LogicalAnd, Indexes...>
  operator&&(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, LogicalAnd, Indexes...>(NULL, *this, Rhs);
  }

  template <typename T, typename = EnableIfScalarType<T>>
  NewRelOp<GetScalarOp<T>, LogicalOr, Indexes...>
  operator||(const T &Rhs) const {
    return NewRelOp<GetScalarOp<T>, LogicalOr, Indexes...>(NULL, *this,
                                                           GetScalarOp<T>(Rhs));
  }

  template <typename RhsOperation,
            typename = EnableIfNoScalarType<RhsOperation>>
  NewRelOp<RhsOperation, LogicalOr, Indexes...>
  operator||(const RhsOperation &Rhs) const {
    return NewRelOp<RhsOperation, LogicalOr, Indexes...>(NULL, *this, Rhs);
  }

private:
  static constexpr int get_vec_idx(int idx) {
    int counter = 0;
    int result = -1;
    ((result = counter++ == idx ? Indexes : result), ...);
    return result;
  }

public:
  template <int... swizzleIndexes>
  ConstSwizzle<get_vec_idx(swizzleIndexes)...> swizzle() const {
    return m_Vector;
  }

  template <int... swizzleIndexes>
  Swizzle<get_vec_idx(swizzleIndexes)...> swizzle() {
    return m_Vector;
  }

  // Leave store() interface to automatic conversion to vec<>.
  // Load to vec_t and then assign to swizzle.
  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t offset, multi_ptr<DataT, Space, DecorateAddress> ptr) {
    vec_t Tmp;
    Tmp.load(offset, ptr);
    *this = Tmp;
  }

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, sizeof...(Indexes)> convert() const {
    // First materialize the swizzle to vec_t and then apply convert() to it.
    vec_t Tmp;
    std::array<int, size()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      Tmp[I] = (*m_Vector)[Idxs[I]];
    }
    return Tmp.template convert<convertT, roundingMode>();
  }

  template <typename asT> asT as() const {
    // First materialize the swizzle to vec_t and then apply as() to it.
    vec_t Tmp = *this;
    static_assert((sizeof(Tmp) == sizeof(asT)),
                  "The new SYCL vec type must have the same storage size in "
                  "bytes as this SYCL swizzled vec");
    static_assert(detail::is_vec_v<asT>,
                  "asT must be SYCL vec of a different element type and "
                  "number of elements specified by asT");
    return Tmp.template as<asT>();
  }

private:
  SwizzleOp(const SwizzleOp &Rhs)
      : m_Vector(Rhs.m_Vector), m_LeftOperation(Rhs.m_LeftOperation),
        m_RightOperation(Rhs.m_RightOperation) {}

  SwizzleOp(VecT *Vector, OperationLeftT LeftOperation,
            OperationRightT RightOperation)
      : m_Vector(Vector), m_LeftOperation(LeftOperation),
        m_RightOperation(RightOperation) {}

  SwizzleOp(VecT *Vector) : m_Vector(Vector) {}

  SwizzleOp(SwizzleOp &&Rhs)
      : m_Vector(Rhs.m_Vector), m_LeftOperation(std::move(Rhs.m_LeftOperation)),
        m_RightOperation(std::move(Rhs.m_RightOperation)) {}

  // Either performing CurrentOperation on results of left and right operands
  // or reading values from actual vector. Perform implicit type conversion when
  // the number of elements == 1

  template <int IdxNum = size()>
  CommonDataT getValue(EnableIfOneIndex<IdxNum, size_t> Index) const {
    if (std::is_same_v<OperationCurrentT<DataT>, GetOp<DataT>>) {
      std::array<int, size()> Idxs{Indexes...};
      return (*m_Vector)[Idxs[Index]];
    }
    auto Op = OperationCurrentT<CommonDataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <int IdxNum = size()>
  DataT getValue(EnableIfMultipleIndexes<IdxNum, size_t> Index) const {
    if (std::is_same_v<OperationCurrentT<DataT>, GetOp<DataT>>) {
      std::array<int, size()> Idxs{Indexes...};
      return (*m_Vector)[Idxs[Index]];
    }
    auto Op = OperationCurrentT<DataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <template <typename> class Operation, typename RhsOperation>
  void operatorHelper(const RhsOperation &Rhs) const {
    Operation<DataT> Op;
    std::array<int, size()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      DataT Res = Op((*m_Vector)[Idxs[I]], Rhs.getValue(I));
      (*m_Vector)[Idxs[I]] = Res;
    }
  }

  // fields
  VecT *m_Vector;

  OperationLeftT m_LeftOperation;
  OperationRightT m_RightOperation;

  // friends
  template <typename T1, int T2> friend class sycl::vec;
  template <typename, int> friend class sycl::detail::vec_base;

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class SwizzleOp;
};
///////////////////////// class SwizzleOp /////////////////////////
} // namespace detail
#endif
} // namespace _V1
} // namespace sycl
