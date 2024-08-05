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

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_lists.hpp>  // for vector_basic_list
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/memcpy.hpp>              // for memcpy
#include <sycl/detail/type_list.hpp>           // for is_contained
#include <sycl/detail/type_traits.hpp>         // for is_floating_point
#include <sycl/detail/vector_arith.hpp>
#include <sycl/detail/vector_convert.hpp>      // for convertImpl
#include <sycl/detail/vector_traits.hpp>       // for vector_alignment
#include <sycl/half_type.hpp>                  // for StorageT, half, Vec16...

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
template <typename Vec, typename T, int N, typename = void>
struct ScalarConversionOperatorMixIn {};

template <typename Vec, typename T, int N>
struct ScalarConversionOperatorMixIn<Vec, T, N, std::enable_if_t<N == 1>> {
  operator T() const { return (*static_cast<const Vec *>(this))[0]; }
};

} // namespace detail

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec
    : public detail::vec_arith<DataT, NumElements>,
      public detail::ScalarConversionOperatorMixIn<vec<DataT, NumElements>,
                                                   DataT, NumElements> {

  static_assert(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
                    NumElements == 4 || NumElements == 8 || NumElements == 16,
                "Invalid number of elements for sycl::vec: only 1, 2, 3, 4, 8 "
                "or 16 are supported");
  static_assert(sizeof(bool) == sizeof(uint8_t), "bool size is not 1 byte");

  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#memory-layout-and-alignment
  // It is required by the SPEC to align vec<DataT, 3> with vec<DataT, 4>.
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;

  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  using DataType = std::array<DataT, AdjustedNum>;

#ifdef __SYCL_DEVICE_ONLY__
  using element_type_for_vector_t = typename detail::map_type<
      DataT,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
      std::byte, /*->*/ std::uint8_t, //
#endif
      bool, /*->*/ std::uint8_t,                            //
      sycl::half, /*->*/ sycl::detail::half_impl::StorageT, //
      sycl::ext::oneapi::bfloat16,
      /*->*/ sycl::ext::oneapi::detail::Bfloat16StorageT, //
      char, /*->*/ detail::ConvertToOpenCLType_t<char>,   //
      DataT, /*->*/ DataT                                 //
      >::type;

public:
  // Type used for passing sycl::vec to SPIRV builtins.
  // We can not use ext_vector_type(1) as it's not supported by SPIRV
  // plugins (CTS fails).
  using vector_t =
      typename std::conditional_t<NumElements == 1, element_type_for_vector_t,
                                  element_type_for_vector_t __attribute__((
                                      ext_vector_type(NumElements)))>;

private:
#endif // __SYCL_DEVICE_ONLY__

  static constexpr int getNumElements() { return NumElements; }

  // SizeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <int Counter, int MaxValue, class...>
  struct SizeChecker : std::conditional_t<Counter == MaxValue, std::true_type,
                                          std::false_type> {};

  template <int Counter, int MaxValue, typename DataT_, class... tail>
  struct SizeChecker<Counter, MaxValue, DataT_, tail...>
      : std::conditional_t<Counter + 1 <= MaxValue,
                           SizeChecker<Counter + 1, MaxValue, tail...>,
                           std::false_type> {};

  // Utility trait for creating an std::array from an vector argument.
  template <typename DataT_, typename T, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const vec<T, sizeof...(Is)> &V, std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }
  template <typename DataT_, typename T, int N, typename T2, typename T3,
            template <typename> class T4, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::SwizzleOp<vec<T, N>, T2, T3, T4, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V.getValue(Is))...};
  }
  template <typename DataT_, typename T, int N, typename T2, typename T3,
            template <typename> class T4, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::SwizzleOp<const vec<T, N>, T2, T3, T4, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V.getValue(Is))...};
  }
  template <typename DataT_, typename T, int N>
  static constexpr std::array<DataT_, N>
  FlattenVecArgHelper(const vec<T, N> &A) {
    return VecToArray<DataT_>(A, std::make_index_sequence<N>());
  }
  template <typename DataT_, typename T, int N, typename T2, typename T3,
            template <typename> class T4, int... T5>
  static constexpr std::array<DataT_, sizeof...(T5)> FlattenVecArgHelper(
      const detail::SwizzleOp<vec<T, N>, T2, T3, T4, T5...> &A) {
    return VecToArray<DataT_>(A, std::make_index_sequence<sizeof...(T5)>());
  }
  template <typename DataT_, typename T, int N, typename T2, typename T3,
            template <typename> class T4, int... T5>
  static constexpr std::array<DataT_, sizeof...(T5)> FlattenVecArgHelper(
      const detail::SwizzleOp<const vec<T, N>, T2, T3, T4, T5...> &A) {
    return VecToArray<DataT_>(A, std::make_index_sequence<sizeof...(T5)>());
  }
  template <typename DataT_, typename T>
  static constexpr auto FlattenVecArgHelper(const T &A) {
    // static_cast required to avoid narrowing conversion warning
    // when T = unsigned long int and DataT_ = int.
    return std::array<DataT_, 1>{static_cast<DataT_>(A)};
  }
  template <typename DataT_, typename T> struct FlattenVecArg {
    constexpr auto operator()(const T &A) const {
      return FlattenVecArgHelper<DataT_>(A);
    }
  };

  // Alias for shortening the vec arguments to array converter.
  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator =
      detail::ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

#define __SYCL_ALLOW_VECTOR_SIZES(num_elements)                                \
  template <int Counter, int MaxValue, typename DataT_, class... tail>         \
  struct SizeChecker<Counter, MaxValue, vec<DataT_, num_elements>, tail...>    \
      : std::conditional_t<                                                    \
            Counter + (num_elements) <= MaxValue,                              \
            SizeChecker<Counter + (num_elements), MaxValue, tail...>,          \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, typename T2,           \
            typename T3, template <typename> class T4, int... T5,              \
            class... tail>                                                     \
  struct SizeChecker<                                                          \
      Counter, MaxValue,                                                       \
      detail::SwizzleOp<vec<DataT_, num_elements>, T2, T3, T4, T5...>,         \
      tail...>                                                                 \
      : std::conditional_t<                                                    \
            Counter + sizeof...(T5) <= MaxValue,                               \
            SizeChecker<Counter + sizeof...(T5), MaxValue, tail...>,           \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, typename T2,           \
            typename T3, template <typename> class T4, int... T5,              \
            class... tail>                                                     \
  struct SizeChecker<                                                          \
      Counter, MaxValue,                                                       \
      detail::SwizzleOp<const vec<DataT_, num_elements>, T2, T3, T4, T5...>,   \
      tail...>                                                                 \
      : std::conditional_t<                                                    \
            Counter + sizeof...(T5) <= MaxValue,                               \
            SizeChecker<Counter + sizeof...(T5), MaxValue, tail...>,           \
            std::false_type> {};

  __SYCL_ALLOW_VECTOR_SIZES(1)
  __SYCL_ALLOW_VECTOR_SIZES(2)
  __SYCL_ALLOW_VECTOR_SIZES(3)
  __SYCL_ALLOW_VECTOR_SIZES(4)
  __SYCL_ALLOW_VECTOR_SIZES(8)
  __SYCL_ALLOW_VECTOR_SIZES(16)
#undef __SYCL_ALLOW_VECTOR_SIZES

  // TypeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};
#define __SYCL_ALLOW_VECTOR_TYPES(num_elements)                                \
  template <typename DataT_>                                                   \
  struct TypeChecker<vec<DataT_, num_elements>, DataT_> : std::true_type {};   \
  template <typename DataT_, typename T2, typename T3,                         \
            template <typename> class T4, int... T5>                           \
  struct TypeChecker<                                                          \
      detail::SwizzleOp<vec<DataT_, num_elements>, T2, T3, T4, T5...>, DataT_> \
      : std::true_type {};                                                     \
  template <typename DataT_, typename T2, typename T3,                         \
            template <typename> class T4, int... T5>                           \
  struct TypeChecker<                                                          \
      detail::SwizzleOp<const vec<DataT_, num_elements>, T2, T3, T4, T5...>,   \
      DataT_> : std::true_type {};

  __SYCL_ALLOW_VECTOR_TYPES(1)
  __SYCL_ALLOW_VECTOR_TYPES(2)
  __SYCL_ALLOW_VECTOR_TYPES(3)
  __SYCL_ALLOW_VECTOR_TYPES(4)
  __SYCL_ALLOW_VECTOR_TYPES(8)
  __SYCL_ALLOW_VECTOR_TYPES(16)
#undef __SYCL_ALLOW_VECTOR_TYPES

  template <int... Indexes>
  using Swizzle =
      detail::SwizzleOp<vec, detail::GetOp<DataT>, detail::GetOp<DataT>,
                        detail::GetOp, Indexes...>;

  template <int... Indexes>
  using ConstSwizzle =
      detail::SwizzleOp<const vec, detail::GetOp<DataT>, detail::GetOp<DataT>,
                        detail::GetOp, Indexes...>;

  // Shortcuts for args validation in vec(const argTN &... args) ctor.
  template <typename... argTN>
  using EnableIfSuitableTypes = typename std::enable_if_t<
      std::conjunction_v<TypeChecker<argTN, DataT>...>>;

  template <typename... argTN>
  using EnableIfSuitableNumElements =
      typename std::enable_if_t<SizeChecker<0, NumElements, argTN...>::value>;

  // Element type for relational operator return value.
  using rel_t = detail::select_cl_scalar_integral_signed_t<DataT>;

public:
  // Aliases required by SYCL 2020 to make sycl::vec consistent
  // with that of marray and buffer.
  using element_type = DataT;
  using value_type = DataT;

  /****************** Constructors **************/
  vec() = default;
  constexpr vec(const vec &Rhs) = default;
  constexpr vec(vec &&Rhs) = default;

private:
  // Implementation detail for the next public ctor.
  template <size_t... Is>
  constexpr vec(const std::array<DataT, NumElements> &Arr,
                std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

public:
  explicit constexpr vec(const DataT &arg)
      : vec{detail::RepeatValue<NumElements>(arg),
            std::make_index_sequence<NumElements>()} {}

  // Constructor from values of base type or vec of base type. Checks that
  // base types are match and that the NumElements == sum of lengths of args.
  template <typename... argTN, typename = EnableIfSuitableTypes<argTN...>,
            typename = EnableIfSuitableNumElements<argTN...>>
  constexpr vec(const argTN &...args)
      : vec{VecArgArrayCreator<DataT, argTN...>::Create(args...),
            std::make_index_sequence<NumElements>()} {}

  /****************** Assignment Operators **************/
  constexpr vec &operator=(const vec &Rhs) = default;

  // Template required to prevent ambiguous overload with the copy assignment
  // when NumElements == 1. The template prevents implicit conversion from
  // vec<_, 1> to DataT.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      std::is_fundamental_v<Ty> ||
          detail::is_half_or_bf16_v<typename std::remove_const_t<Ty>>,
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

#ifdef __SYCL_DEVICE_ONLY__
  // Make it a template to avoid ambiguity with `vec(const DataT &)` when
  // `vector_t` is the same as `DataT`. Not that the other ctor isn't a template
  // so we don't even need a smart `enable_if` condition here, the mere fact of
  // this being a template makes the other ctor preferred.
  template <
      typename vector_t_ = vector_t,
      typename = typename std::enable_if_t<std::is_same_v<vector_t_, vector_t>>>
  constexpr vec(vector_t_ openclVector) {
    m_Data = sycl::bit_cast<DataType>(openclVector);
  }

  /* @SYCL2020
   * Available only when: compiled for the device.
   * Converts this SYCL vec instance to the underlying backend-native vector
   * type defined by vector_t.
   */
  operator vector_t() const { return sycl::bit_cast<vector_t>(m_Data); }
#endif // __SYCL_DEVICE_ONLY__

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(m_Data); }

private:
  // We interpret bool as int8_t, std::byte as uint8_t for conversion to other
  // types.
  template <typename T>
  using ConvertBoolAndByteT =
      typename detail::map_type<T,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
                                std::byte, /*->*/ std::uint8_t, //
#endif
                                bool, /*->*/ std::uint8_t, //
                                T, /*->*/ T                //
                                >::type;

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
      return sycl::bit_cast<RetType>(m_Data[Index]);
    else
#endif
      return static_cast<RetType>(m_Data[Index]);
  }

public:
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const {

    using T = ConvertBoolAndByteT<DataT>;
    using R = ConvertBoolAndByteT<convertT>;
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    static_assert(std::is_integral_v<R> ||
                      detail::is_floating_point<R>::value ||
                      std::is_same_v<R, bfloat16>,
                  "Unsupported convertT");

    using OpenCLT = detail::ConvertToOpenCLType_t<T>;
    using OpenCLR = detail::ConvertToOpenCLType_t<R>;
    vec<convertT, NumElements> Result;

    // convertImpl can't be called with the same From and To types and therefore
    // we need some special processing in a few cases.
    if constexpr (std::is_same_v<DataT, convertT>) {
      return *this;
    } else if constexpr (std::is_same_v<OpenCLT, OpenCLR> ||
                         std::is_same_v<T, R>) {
      for (size_t I = 0; I < NumElements; ++I)
        Result[I] = static_cast<convertT>(getValue(I));
      return Result;
    } else {

#ifdef __SYCL_DEVICE_ONLY__
      using OpenCLVecT = OpenCLT __attribute__((ext_vector_type(NumElements)));
      using OpenCLVecR = OpenCLR __attribute__((ext_vector_type(NumElements)));

      auto NativeVector = sycl::bit_cast<vector_t>(*this);
      using ConvertTVecType = typename vec<convertT, NumElements>::vector_t;

      // Whole vector conversion can only be done, if:
      constexpr bool canUseNativeVectorConvert =
#ifdef __NVPTX__
          //  TODO: Likely unnecessary as
          //  https://github.com/intel/llvm/issues/11840 has been closed
          //  already.
          false &&
#endif
          NumElements > 1 &&
          // - vec storage has an equivalent OpenCL native vector it is
          //   implicitly convertible to. There are some corner cases where it
          //   is not the case with char, long and long long types.
          std::is_convertible_v<vector_t, OpenCLVecT> &&
          std::is_convertible_v<ConvertTVecType, OpenCLVecR> &&
          // - it is not a signed to unsigned (or vice versa) conversion
          //   see comments within 'convertImpl' for more details;
          !detail::is_sint_to_from_uint<T, R>::value &&
          // - destination type is not bool. bool is stored as integer under the
          //   hood and therefore conversion to bool looks like conversion
          //   between two integer types. Since bit pattern for true and false
          //   is not defined, there is no guarantee that integer conversion
          //   yields right results here;
          !std::is_same_v<convertT, bool>;

      if constexpr (canUseNativeVectorConvert) {
        auto val = detail::convertImpl<T, R, roundingMode, NumElements, OpenCLVecT,
                                OpenCLVecR>(NativeVector);
        Result.m_Data = sycl::bit_cast<decltype(Result.m_Data)>(val);
      } else
#endif // __SYCL_DEVICE_ONLY__
      {
        // Otherwise, we fallback to per-element conversion:
        for (size_t I = 0; I < NumElements; ++I) {
          auto val =
              detail::convertImpl<T, R, roundingMode, 1, OpenCLT, OpenCLR>(
                  getValue(I));
#ifdef __SYCL_DEVICE_ONLY__
          // On device, we interpret BF16 as uint16.
          if constexpr (std::is_same_v<convertT, bfloat16>)
            Result[I] = sycl::bit_cast<convertT>(val);
          else
#endif
            Result[I] = static_cast<convertT>(val);
        }
      }
    }
    return Result;
  }

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
    return this;
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
    return this;
  }

  const DataT &operator[](int i) const { return m_Data[i]; }

  DataT &operator[](int i) { return m_Data[i]; }

  // Begin hi/lo, even/odd, xyzw, and rgba swizzles. @{
private:
  // Indexer used in the swizzles.def
  // Currently it is defined as a template struct. Replacing it with a constexpr
  // function would activate a bug in MSVC that is fixed only in v19.20.
  // Until then MSVC does not recognize such constexpr functions as const and
  // thus does not let using them in template parameters inside swizzle.def.
  template <int Index> struct Indexer {
    static constexpr int value = Index;
  };

public:
#ifdef __SYCL_ACCESS_RETURN
#error "Undefine __SYCL_ACCESS_RETURN macro"
#endif
#define __SYCL_ACCESS_RETURN this
#include "swizzles.def"
#undef __SYCL_ACCESS_RETURN
  // }@ End of hi/lo, even/odd, xyzw, and rgba swizzles.

  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    for (int I = 0; I < NumElements; I++) {
      m_Data[I] = *multi_ptr<const DataT, Space, DecorateAddress>(
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
      m_Data[I] = Ptr[Offset * NumElements + I];
  }

  template <access::address_space Space, access::decorated DecorateAddress>
  void store(size_t Offset,
             multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    for (int I = 0; I < NumElements; I++) {
      *multi_ptr<DataT, Space, DecorateAddress>(Ptr + Offset * NumElements +
                                                I) = m_Data[I];
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
      Ptr[Offset * NumElements + I] = m_Data[I];
  }

private:
  // fields
  // Alignment is the same as size, to a maximum size of 64. SPEC requires
  // "The elements of an instance of the SYCL vec class template are stored
  // in memory sequentially and contiguously and are aligned to the size of
  // the element type in bytes multiplied by the number of elements."
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

  // friends
  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class detail::SwizzleOp;
  template <typename T1, int T2> friend class __SYCL_EBO vec;
  // To allow arithmetic operators access private members of vec.
  template <typename T1, int T2> friend class detail::vec_arith;
  template <typename T1, int T2> friend class detail::vec_arith_common;
};
///////////////////////// class sycl::vec /////////////////////////

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = std::enable_if_t<(std::is_same_v<T, U> && ...)>>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;
#endif

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
template <typename T>
using rel_t = detail::select_cl_scalar_integral_signed_t<T>;

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
class SwizzleOp {
  using DataT = typename VecT::element_type;
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
  static constexpr int getNumElements() { return sizeof...(Indexes); }

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
  using NewRelOp = SwizzleOp<vec<rel_t, VecT::getNumElements()>,
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
  using EnableIfOneIndex = typename std::enable_if_t<
      1 == IdxNum && SwizzleOp::getNumElements() == IdxNum, T>;

  template <int IdxNum, typename T = void>
  using EnableIfMultipleIndexes = typename std::enable_if_t<
      1 != IdxNum && SwizzleOp::getNumElements() == IdxNum, T>;

  template <typename T>
  using EnableIfScalarType = typename std::enable_if_t<
      std::is_convertible_v<DataT, T> &&
      (std::is_fundamental_v<T> ||
       detail::is_half_or_bf16_v<typename std::remove_const_t<T>>)>;

  template <typename T>
  using EnableIfNoScalarType = typename std::enable_if_t<
      !std::is_convertible_v<DataT, T> ||
      !(std::is_fundamental_v<T> ||
        detail::is_half_or_bf16_v<typename std::remove_const_t<T>>)>;

  template <int... Indices>
  using Swizzle =
      SwizzleOp<VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

  template <int... Indices>
  using ConstSwizzle =
      SwizzleOp<const VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

public:
  using element_type = DataT;
  using value_type = DataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = typename vec_t::vector_t;
#endif // __SYCL_DEVICE_ONLY__

  const DataT &operator[](int i) const {
    std::array<int, getNumElements()> Idxs{Indexes...};
    return (*m_Vector)[Idxs[i]];
  }

  template <typename _T = VecT>
  std::enable_if_t<!std::is_const_v<_T>, DataT> &operator[](int i) {
    std::array<int, getNumElements()> Idxs{Indexes...};
    return (*m_Vector)[Idxs[i]];
  }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  static constexpr size_t size() noexcept { return getNumElements(); }

  template <int Num = getNumElements()>
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const {
    return byte_size<Num>();
  }

  template <int Num = getNumElements()> size_t byte_size() const noexcept {
    return sizeof(DataT) * (Num == 3 ? 4 : Num);
  }

  template <typename T, int IdxNum = getNumElements(),
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

  template <int IdxNum = getNumElements(),
            typename = EnableIfMultipleIndexes<IdxNum>>
  SwizzleOp &operator=(const vec<DataT, IdxNum> &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      (*m_Vector)[Idxs[I]] = Rhs[I];
    }
    return *this;
  }

  template <int IdxNum = getNumElements(), typename = EnableIfOneIndex<IdxNum>>
  SwizzleOp &operator=(const DataT &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    (*m_Vector)[Idxs[0]] = Rhs;
    return *this;
  }

  template <int IdxNum = getNumElements(),
            EnableIfMultipleIndexes<IdxNum, bool> = true>
  SwizzleOp &operator=(const DataT &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    for (auto Idx : Idxs) {
      (*m_Vector)[Idx] = Rhs;
    }
    return *this;
  }

  template <int IdxNum = getNumElements(), typename = EnableIfOneIndex<IdxNum>>
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

  template <
      typename T1, typename T2, typename T3, template <typename> class T4,
      int... T5,
      typename = typename std::enable_if_t<sizeof...(T5) == getNumElements()>>
  SwizzleOp &operator=(const SwizzleOp<T1, T2, T3, T4, T5...> &Rhs) {
    std::array<int, getNumElements()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      (*m_Vector)[Idxs[I]] = Rhs.getValue(I);
    }
    return *this;
  }

  template <
      typename T1, typename T2, typename T3, template <typename> class T4,
      int... T5,
      typename = typename std::enable_if_t<sizeof...(T5) == getNumElements()>>
  SwizzleOp &operator=(SwizzleOp<T1, T2, T3, T4, T5...> &&Rhs) {
    std::array<int, getNumElements()> Idxs{Indexes...};
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

  // Begin hi/lo, even/odd, xyzw, and rgba swizzles.
private:
  // Indexer used in the swizzles.def.
  // Currently it is defined as a template struct. Replacing it with a constexpr
  // function would activate a bug in MSVC that is fixed only in v19.20.
  // Until then MSVC does not recognize such constexpr functions as const and
  // thus does not let using them in template parameters inside swizzle.def.
  template <int Index> struct Indexer {
    static constexpr int IDXs[sizeof...(Indexes)] = {Indexes...};
    static constexpr int value = IDXs[Index >= getNumElements() ? 0 : Index];
  };

public:
#ifdef __SYCL_ACCESS_RETURN
#error "Undefine __SYCL_ACCESS_RETURN macro"
#endif
#define __SYCL_ACCESS_RETURN m_Vector
#include "swizzles.def"
#undef __SYCL_ACCESS_RETURN
  // End of hi/lo, even/odd, xyzw, and rgba swizzles.

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
    std::array<int, getNumElements()> Idxs{Indexes...};
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
    static_assert(
        detail::is_contained<asT, detail::gtl::vector_basic_list>::value ||
            detail::is_contained<asT, detail::gtl::vector_bool_list>::value,
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

  template <int IdxNum = getNumElements()>
  CommonDataT getValue(EnableIfOneIndex<IdxNum, size_t> Index) const {
    if (std::is_same_v<OperationCurrentT<DataT>, GetOp<DataT>>) {
      std::array<int, getNumElements()> Idxs{Indexes...};
      return (*m_Vector)[Idxs[Index]];
    }
    auto Op = OperationCurrentT<CommonDataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <int IdxNum = getNumElements()>
  DataT getValue(EnableIfMultipleIndexes<IdxNum, size_t> Index) const {
    if (std::is_same_v<OperationCurrentT<DataT>, GetOp<DataT>>) {
      std::array<int, getNumElements()> Idxs{Indexes...};
      return (*m_Vector)[Idxs[Index]];
    }
    auto Op = OperationCurrentT<DataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <template <typename> class Operation, typename RhsOperation>
  void operatorHelper(const RhsOperation &Rhs) const {
    Operation<DataT> Op;
    std::array<int, getNumElements()> Idxs{Indexes...};
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

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class SwizzleOp;
};
///////////////////////// class SwizzleOp /////////////////////////
} // namespace detail
} // namespace _V1
} // namespace sycl
