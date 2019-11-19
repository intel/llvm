//==---------------- types.hpp --- SYCL types ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements vec and __swizzled_vec__ classes.

#pragma once

// Define __NO_EXT_VECTOR_TYPE_ON_HOST__ to avoid using ext_vector_type
// extension even if the host compiler supports it. The same can be
// accomplished by -D__NO_EXT_VECTOR_TYPE_ON_HOST__ command line option.
#ifndef __NO_EXT_VECTOR_TYPE_ON_HOST__
// #define __NO_EXT_VECTOR_TYPE_ON_HOST__
#endif

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

#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
#error "Undefine __SYCL_USE_EXT_VECTOR_TYPE__ macro"
#endif
#ifdef __HAS_EXT_VECTOR_TYPE__
#if defined(__SYCL_DEVICE_ONLY__) || !defined(__NO_EXT_VECTOR_TYPE_ON_HOST__)
#define __SYCL_USE_EXT_VECTOR_TYPE__
#endif
#elif defined(__SYCL_DEVICE_ONLY__)
// This is a soft error. We expect the device compiler to have ext_vector_type
// support, but that should not be a hard requirement.
#error "SYCL device compiler is built without ext_vector_type support"
#endif // __HAS_EXT_VECTOR_TYPE__

#include <CL/sycl/aliases.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/half_type.hpp>
#include <CL/sycl/multi_ptr.hpp>

#include <array>
#include <cmath>
#include <cstring>
#ifndef __SYCL_DEVICE_ONLY__
#include <cfenv>
#endif

// 4.10.1: Scalar data types
// 4.10.2: SYCL vector types

namespace cl {
namespace sycl {

enum class rounding_mode { automatic, rte, rtz, rtp, rtn };
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

template <typename T, int N> class BaseCLTypeConverter;

// Element type for relational operator return value.
template <typename DataT>
using rel_t = typename std::conditional<
    sizeof(DataT) == sizeof(cl_char), cl_char,
    typename std::conditional<
        sizeof(DataT) == sizeof(cl_short), cl_short,
        typename std::conditional<
            sizeof(DataT) == sizeof(cl_int), cl_int,
            typename std::conditional<sizeof(DataT) == sizeof(cl_long), cl_long,
                                      bool>::type>::type>::type>::type;

// Special type indicating that SwizzleOp should just read value from vector -
// not trying to perform any operations. Should not be called.
template <typename T> class GetOp {
public:
  using DataT = T;
  DataT getValue(size_t Index) const { return 0; }
  DataT operator()(DataT LHS, DataT Rhs) { return 0; }
};

// Special type for working SwizzleOp with scalars, stores a scalar and gives
// the scalar at any index. Provides interface is compatible with SwizzleOp
// operations
template <typename T> class GetScalarOp {
public:
  using DataT = T;
  GetScalarOp(DataT Data) : m_Data(Data) {}
  DataT getValue(size_t Index) const { return m_Data; }

private:
  DataT m_Data;
};

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

template <typename T, typename R>
using is_int_to_int =
    std::integral_constant<bool, std::is_integral<T>::value &&
                                     std::is_integral<R>::value>;

template <typename T, typename R>
using is_int_to_float =
    std::integral_constant<bool, std::is_integral<T>::value &&
                                     detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_float_to_int =
    std::integral_constant<bool, detail::is_floating_point<T>::value &&
                                     std::is_integral<R>::value>;

template <typename T, typename R>
using is_float_to_float =
    std::integral_constant<bool, detail::is_floating_point<T>::value &&
                                     detail::is_floating_point<R>::value>;

template <typename T, typename R, rounding_mode roundingMode>
detail::enable_if_t<std::is_same<T, R>::value, R> convertImpl(T Value) {
  return Value;
}

// Note for float to half conversions, static_cast calls the conversion operator
// implemented for host that takes care of the precision requirements.
template <typename T, typename R, rounding_mode roundingMode>
detail::enable_if_t<!std::is_same<T, R>::value &&
                        (is_int_to_int<T, R>::value ||
                         is_int_to_float<T, R>::value ||
                         is_float_to_float<T, R>::value),
                    R>
convertImpl(T Value) {
  return static_cast<R>(Value);
}

// float to int
template <typename T, typename R, rounding_mode roundingMode>
detail::enable_if_t<is_float_to_int<T, R>::value, R> convertImpl(T Value) {
#ifndef __SYCL_DEVICE_ONLY__
  switch (roundingMode) {
    // Round to nearest even is default rounding mode for floating-point types
  case rounding_mode::automatic:
    // Round to nearest even.
  case rounding_mode::rte: {
    int OldRoundingDirection = std::fegetround();
    int Err = std::fesetround(FE_TONEAREST);
    if (Err)
      throw runtime_error("Unable to set rounding mode to FE_TONEAREST");
    R Result = std::rint(Value);
    Err = std::fesetround(OldRoundingDirection);
    if (Err)
      throw runtime_error("Unable to restore rounding mode.");
    return Result;
  }
    // Round toward zero.
  case rounding_mode::rtz:
    return std::trunc(Value);
    // Round toward positive infinity.
  case rounding_mode::rtp:
    return std::ceil(Value);
    // Round toward negative infinity.
  case rounding_mode::rtn:
    return std::floor(Value);
  default:
    assert(!"Unsupported rounding mode!");
    return static_cast<R>(Value);
  };
#else
  // TODO implement device side conversion.
  return static_cast<R>(Value);
#endif
}

// 4.10.2.6 Memory layout and alignment
template <int N> struct VectorLength { constexpr static int value = N; };

template <> struct VectorLength<3> { constexpr static int value = 4; };

template <typename T, int N> struct VectorAlignment {
  constexpr static int value = sizeof(T) * VectorLength<N>::value;
};

} // namespace detail

#if defined(_WIN32) && (_MSC_VER)
// MSVC Compiler doesn't allow using of function arguments with alignment
// requirements. MSVC Compiler Error C2719: 'parameter': formal parameter with
// __declspec(align('#')) won't be aligned. The align __declspec modifier
// is not permitted on function parameters. Function parameter alignment
// is controlled by the calling convention used.
// For more information, see Calling Conventions
// (https://docs.microsoft.com/en-us/cpp/cpp/calling-conventions).
// For information on calling conventions for x64 processors, see
// Calling Convention
// (https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention).
#pragma message ("Alignment of class vec is not in accordance with SYCL \
specification requirements, a limitation of the MSVC compiler(Error C2719).\
Applied default alignment.")
#define SYCL_ALIGNAS(x)
#else
#define SYCL_ALIGNAS(N) alignas(N)
#endif

template <typename Type, int NumElements> class vec {
  using DataT = Type;

  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  using DataType =
      typename detail::BaseCLTypeConverter<DataT, NumElements>::DataType;

  static constexpr int getNumElements() { return NumElements; }

  // SizeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <int Counter, int MaxValue, class...>
  struct SizeChecker: detail::conditional_t<Counter == MaxValue,
      std::true_type, std::false_type> {};

  template <int Counter, int MaxValue, typename DataT_, class... tail>
  struct SizeChecker<Counter, MaxValue, DataT_, tail...>
      : detail::conditional_t<Counter + 1 <= MaxValue,
                      SizeChecker<Counter + 1, MaxValue, tail...>,
                      std::false_type> {};

#define ALLOW_VECTOR_SIZES(num_elements)                                       \
  template <int Counter, int MaxValue, typename DataT_, class... tail>         \
  struct SizeChecker<Counter, MaxValue, vec<DataT_, num_elements>, tail...>    \
      : detail::conditional_t<                                                 \
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
      : detail::conditional_t<                                                 \
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
      : detail::conditional_t<                                                 \
            Counter + sizeof...(T5) <= MaxValue,                               \
            SizeChecker<Counter + sizeof...(T5), MaxValue, tail...>,           \
            std::false_type> {};

  ALLOW_VECTOR_SIZES(1)
  ALLOW_VECTOR_SIZES(2)
  ALLOW_VECTOR_SIZES(3)
  ALLOW_VECTOR_SIZES(4)
  ALLOW_VECTOR_SIZES(8)
  ALLOW_VECTOR_SIZES(16)
#undef ALLOW_VECTOR_SIZES

  template <class...> struct conjunction : std::true_type {};
  template <class B1, class... tail>
  struct conjunction<B1, tail...>
      : detail::conditional_t<bool(B1::value), conjunction<tail...>, B1> {};

  // TypeChecker is needed for vec(const argTN &... args) ctor to validate args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};
#define ALLOW_VECTOR_TYPES(num_elements)                                       \
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

  ALLOW_VECTOR_TYPES(1)
  ALLOW_VECTOR_TYPES(2)
  ALLOW_VECTOR_TYPES(3)
  ALLOW_VECTOR_TYPES(4)
  ALLOW_VECTOR_TYPES(8)
  ALLOW_VECTOR_TYPES(16)
#undef ALLOW_VECTOR_TYPES

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
  using EnableIfSuitableTypes = typename std::enable_if<
      conjunction<TypeChecker<argTN, DataT>...>::value>::type;

  template <typename... argTN>
  using EnableIfSuitableNumElements = typename std::enable_if<
      SizeChecker<0, NumElements, argTN...>::value>::type;

public:
  using element_type = DataT;
  using rel_t = detail::rel_t<DataT>;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = DataType;
#endif

  vec() = default;

  // TODO Remove this difference between host and device side after
  // when root cause of API incompatibility will be fixed
#ifdef __SYCL_DEVICE_ONLY__
  vec(const vec &Rhs) = default;
#else
  vec(const vec &Rhs) : m_Data(Rhs.m_Data) {}
#endif

  vec(vec &&Rhs) = default;

  vec &operator=(const vec &Rhs) = default;

  // W/o this, things like "vec<char,*> = vec<signed char, *>" doesn't work.
  template <typename Ty = DataT>
  typename std::enable_if<!std::is_same<Ty, rel_t>::value &&
                              std::is_convertible<Ty, rel_t>::value,
                          vec &>::type
  operator=(const vec<rel_t, NumElements> &Rhs) {
    *this = Rhs.template as<vec>();
    return *this;
  }
  
#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
  template <typename T = void>
  using EnableIfNotHostHalf = typename std::enable_if<
      !std::is_same<DataT, cl::sycl::detail::half_impl::half>::value, T>::type;
  template <typename T = void>
  using EnableIfHostHalf = typename std::enable_if<
      std::is_same<DataT, cl::sycl::detail::half_impl::half>::value, T>::type;

  template <typename Ty = DataT>
  explicit vec(const EnableIfNotHostHalf<Ty> &arg) {
    m_Data = (DataType)arg;
  }

  template <typename Ty = DataT>
  typename std::enable_if<
      std::is_fundamental<Ty>::value ||
          std::is_same<typename std::remove_const<Ty>::type, half>::value,
      vec &>::type
  operator=(const EnableIfNotHostHalf<Ty> &Rhs) {
    m_Data = (DataType)Rhs;
    return *this;
  }

  template <typename Ty = DataT> explicit vec(const EnableIfHostHalf<Ty> &arg) {
    for (int i = 0; i < NumElements; ++i) {
      setValue(i, arg);
    }
  }

  template <typename Ty = DataT>
  typename std::enable_if<
      std::is_fundamental<Ty>::value ||
          std::is_same<typename std::remove_const<Ty>::type, half>::value,
      vec &>::type
  operator=(const EnableIfHostHalf<Ty> &Rhs) {
    for (int i = 0; i < NumElements; ++i) {
      setValue(i, Rhs);
    }
    return *this;
  }
#else
  explicit vec(const DataT &arg) {
    for (int i = 0; i < NumElements; ++i) {
      setValue(i, arg);
    }
  }

  template <typename Ty = DataT>
  typename std::enable_if<
      std::is_fundamental<Ty>::value ||
          std::is_same<typename std::remove_const<Ty>::type, half>::value,
      vec &>::type
  operator=(const DataT &Rhs) {
    for (int i = 0; i < NumElements; ++i) {
      setValue(i, Rhs);
    }
    return *this;
  }
#endif

#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
  // Optimized naive constructors with NumElements of DataT values.
  // We don't expect compilers to optimize vararg recursive functions well.

  // Helper type to make specific constructors available only for specific
  // number of elements.
  template <int IdxNum, typename T = void>
  using EnableIfMultipleElems = typename std::enable_if<
      std::is_convertible<T, DataT>::value && NumElements == IdxNum,
      DataT>::type;
  template <typename Ty = DataT>
  vec(const EnableIfMultipleElems<2, Ty> Arg0,
      const EnableIfNotHostHalf<Ty> Arg1)
      : m_Data{Arg0, Arg1} {}
  template <typename Ty = DataT>
  vec(const EnableIfMultipleElems<3, Ty> Arg0,
      const EnableIfNotHostHalf<Ty> Arg1, const DataT Arg2)
      : m_Data{Arg0, Arg1, Arg2} {}
  template <typename Ty = DataT>
  vec(const EnableIfMultipleElems<4, Ty> Arg0,
      const EnableIfNotHostHalf<Ty> Arg1, const DataT Arg2, const Ty Arg3)
      : m_Data{Arg0, Arg1, Arg2, Arg3} {}
  template <typename Ty = DataT>
  vec(const EnableIfMultipleElems<8, Ty> Arg0,
      const EnableIfNotHostHalf<Ty> Arg1, const DataT Arg2, const DataT Arg3,
      const DataT Arg4, const DataT Arg5, const DataT Arg6, const DataT Arg7)
      : m_Data{Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7} {}
  template <typename Ty = DataT>
  vec(const EnableIfMultipleElems<16, Ty> Arg0,
      const EnableIfNotHostHalf<Ty> Arg1, const DataT Arg2, const DataT Arg3,
      const DataT Arg4, const DataT Arg5, const DataT Arg6, const DataT Arg7,
      const DataT Arg8, const DataT Arg9, const DataT ArgA, const DataT ArgB,
      const DataT ArgC, const DataT ArgD, const DataT ArgE, const DataT ArgF)
      : m_Data{Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7,
               Arg8, Arg9, ArgA, ArgB, ArgC, ArgD, ArgE, ArgF} {}
#endif

  // Constructor from values of base type or vec of base type. Checks that
  // base types are match and that the NumElements == sum of lengths of args.
  template <typename... argTN, typename = EnableIfSuitableTypes<argTN...>,
            typename = EnableIfSuitableNumElements<argTN...>>
  vec(const argTN &... args) {
    vaargCtorHelper(0, args...);
  }

  // TODO: Remove, for debug purposes only.
  void dump() {
#ifndef __SYCL_DEVICE_ONLY__
    for (int I = 0; I < NumElements; ++I) {
      std::cout << "  " << I << ": " << getValue(I) << std::endl;
    }
    std::cout << std::endl;
#endif // __SYCL_DEVICE_ONLY__
  }

#ifdef __SYCL_DEVICE_ONLY__
  template <typename vector_t_ = vector_t,
            typename = typename std::enable_if<
                std::is_same<vector_t_, vector_t>::value &&
                !std::is_same<vector_t_, DataT>::value>::type>
  vec(vector_t openclVector) : m_Data(openclVector) {}
  operator vector_t() const { return m_Data; }
#endif

  // Available only when: NumElements == 1
  template <int N = NumElements>
  operator typename std::enable_if<N == 1, DataT>::type() const {
    return m_Data;
  }
  static constexpr size_t get_count() { return NumElements; }
  static constexpr size_t get_size() { return sizeof(m_Data); }

  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const {
    static_assert(std::is_integral<convertT>::value ||
                      detail::is_floating_point<convertT>::value,
                  "Unsupported convertT");
    vec<convertT, NumElements> Result;
    for (size_t I = 0; I < NumElements; ++I) {
      Result.setValue(
          I, detail::convertImpl<DataT, convertT, roundingMode>(getValue(I)));
    }
    return Result;
  }

  template <typename asT> asT as() const {
    static_assert((sizeof(*this) == sizeof(asT)),
                  "The new SYCL vec type must have the same storage size in "
                  "bytes as this SYCL vec");
    static_assert(
        detail::is_contained<asT, detail::gtl::vector_basic_list>::value,
        "asT must be SYCL vec of a different element type and "
        "number of elements specified by asT");
    asT Result;
    std::memcpy(&Result.m_Data, &m_Data, sizeof(decltype(Result.m_Data)));
    return Result;
  }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
    return this;
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
    return this;
  }

  // ext_vector_type is used as an underlying type for sycl::vec on device.
  // The problem is that for clang vector types the return of operator[] is a
  // temporary and not a reference to the element in the vector. In practice
  // reinterpret_cast<DataT *>(&m_Data)[i]; is working. According to
  // http://llvm.org/docs/GetElementPtr.html#can-gep-index-into-vector-elements
  // this is not disallowed now. But could probably be disallowed in the future.
  // That is why tests are added to check that behavior of the compiler has
  // not changed.
  //
  // Implement operator [] in the same way for host and device.
  // TODO: change host side implementation when underlying type for host side
  // will be changed to std::array.
  const DataT &operator[](int i) const {
    return reinterpret_cast<const DataT *>(&m_Data)[i];
  }

  DataT &operator[](int i) { return reinterpret_cast<DataT *>(&m_Data)[i]; }

  // Begin hi/lo, even/odd, xyzw, and rgba swizzles.
private:
  // Indexer used in the swizzles.def
  // Currently it is defined as a template struct. Replacing it with a constexpr
  // function would activate a bug in MSVC that is fixed only in v19.20.
  // Until then MSVC does not recognize such constexpr functions as const and
  // thus does not let using them in template parameters inside swizzle.def.
  template <int Index>
  struct Indexer {
    static constexpr int value = Index;
  };

public:
#ifdef __SYCL_ACCESS_RETURN
#error "Undefine __SYCL_ACCESS_RETURN macro"
#endif
#define __SYCL_ACCESS_RETURN this
#include "swizzles.def"
#undef __SYCL_ACCESS_RETURN
  // End of hi/lo, even/odd, xyzw, and rgba swizzles.

  // TODO: make templated address space to work.
  // Somehow, access<> to multi_ptr<> conversion doesn't work w/o making
  // address space explicitly specified.
#ifdef __SYCL_LOADSTORE
#error "Undefine __SYCL_LOADSTORE macro"
#endif
#define __SYCL_LOADSTORE(Space)                                                \
  void load(size_t Offset, multi_ptr<const DataT, Space> Ptr) {                \
    for (int I = 0; I < NumElements; I++) {                                    \
      setValue(                                                                \
          I, *multi_ptr<const DataT, Space>(Ptr + Offset * NumElements + I));  \
    }                                                                          \
  }                                                                            \
  void store(size_t Offset, multi_ptr<DataT, Space> Ptr) const {               \
    for (int I = 0; I < NumElements; I++) {                                    \
      *multi_ptr<DataT, Space>(Ptr + Offset * NumElements + I) = getValue(I);  \
    }                                                                          \
  }

  __SYCL_LOADSTORE(access::address_space::global_space)
  __SYCL_LOADSTORE(access::address_space::local_space)
  __SYCL_LOADSTORE(access::address_space::constant_space)
  __SYCL_LOADSTORE(access::address_space::private_space)
#undef __SYCL_LOADSTORE

#ifdef __SYCL_BINOP
#error "Undefine __SYCL_BINOP macro"
#endif

#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
#define __SYCL_BINOP(BINOP, OPASSIGN)                                          \
  template <typename Ty = vec>                                                 \
  vec operator BINOP(const EnableIfNotHostHalf<Ty> &Rhs) const {               \
    vec Ret;                                                                   \
    Ret.m_Data = m_Data BINOP Rhs.m_Data;                                      \
    return Ret;                                                                \
  }                                                                            \
  template <typename Ty = vec>                                                 \
  vec operator BINOP(const EnableIfHostHalf<Ty> &Rhs) const {                  \
    vec Ret;                                                                   \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret.setValue(I, (getValue(I) BINOP Rhs.getValue(I)));                    \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<                                                     \
      std::is_convertible<DataT, T>::value &&                                  \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      vec>::type                                                               \
  operator BINOP(const T &Rhs) const {                                         \
    return *this BINOP vec(static_cast<const DataT &>(Rhs));                   \
  }                                                                            \
  vec &operator OPASSIGN(const vec &Rhs) {                                     \
    *this = *this BINOP Rhs;                                                   \
    return *this;                                                              \
  }                                                                            \
  template <int Num = NumElements>                                             \
  typename std::enable_if<Num != 1, vec &>::type                               \
  operator OPASSIGN(const DataT &Rhs) {                                        \
    *this = *this BINOP vec(Rhs);                                              \
    return *this;                                                              \
  }
#else // __SYCL_USE_EXT_VECTOR_TYPE__
#define __SYCL_BINOP(BINOP, OPASSIGN)                                          \
  vec operator BINOP(const vec &Rhs) const {                                   \
    vec Ret;                                                                   \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret.setValue(I, (getValue(I) BINOP Rhs.getValue(I)));                    \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<                                                     \
      std::is_convertible<DataT, T>::value &&                                  \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      vec>::type                                                               \
  operator BINOP(const T &Rhs) const {                                         \
    return *this BINOP vec(static_cast<const DataT &>(Rhs));                   \
  }                                                                            \
  vec &operator OPASSIGN(const vec &Rhs) {                                     \
    *this = *this BINOP Rhs;                                                   \
    return *this;                                                              \
  }                                                                            \
  template <int Num = NumElements>                                             \
  typename std::enable_if<Num != 1, vec &>::type operator OPASSIGN(            \
      const DataT &Rhs) {                                                      \
    *this = *this BINOP vec(Rhs);                                              \
    return *this;                                                              \
  }
#endif // __SYCL_USE_EXT_VECTOR_TYPE__

  __SYCL_BINOP(+, +=)
  __SYCL_BINOP(-, -=)
  __SYCL_BINOP(*, *=)
  __SYCL_BINOP(/, /=)

  // TODO: The following OPs are available only when: DataT != cl_float &&
  // DataT != cl_double && DataT != cl_half
  __SYCL_BINOP(%, %=)
  __SYCL_BINOP(|, |=)
  __SYCL_BINOP(&, &=)
  __SYCL_BINOP(^, ^=)
  __SYCL_BINOP(>>, >>=)
  __SYCL_BINOP(<<, <<=)
#undef __SYCL_BINOP
#undef __SYCL_BINOP_HELP

  // Note: vec<>/SwizzleOp logical value is 0/-1 logic, as opposed to 0/1 logic.
  // As far as CTS validation is concerned, 0/-1 logic also applies when
  // NumElements is equal to one, which is somewhat inconsistent with being
  // transparent with scalar data.
  // TODO: Determine if vec<, NumElements=1> is needed at all, remove this
  // inconsistency if not by disallowing one-element vectors (as in OpenCL)

#ifdef __SYCL_RELLOGOP
#error "Undefine __SYCL_RELLOGOP macro"
#endif
// Use __SYCL_DEVICE_ONLY__ macro because cast to OpenCL vector type is defined
// by SYCL device compiler only.
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_RELLOGOP(RELLOGOP)                                              \
  vec<rel_t, NumElements> operator RELLOGOP(const vec &Rhs) const {            \
    auto Ret =                                                                 \
        vec<rel_t, NumElements>((typename vec<rel_t, NumElements>::vector_t)(  \
            m_Data RELLOGOP Rhs.m_Data));                                      \
    if (NumElements == 1) /*Scalar 0/1 logic was applied, invert*/             \
      Ret *= -1;                                                               \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<std::is_convertible<T, DataT>::value &&              \
                              (std::is_fundamental<T>::value ||                \
                               std::is_same<T, half>::value),                  \
                          vec<rel_t, NumElements>>::type                       \
  operator RELLOGOP(const T &Rhs) const {                                      \
    return *this RELLOGOP vec(static_cast<const DataT &>(Rhs));                \
  }
#else
#define __SYCL_RELLOGOP(RELLOGOP)                                              \
  vec<rel_t, NumElements> operator RELLOGOP(const vec &Rhs) const {            \
    vec<rel_t, NumElements> Ret;                                               \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret.setValue(I, -(getValue(I) RELLOGOP Rhs.getValue(I)));                \
    }                                                                          \
    return Ret;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  typename std::enable_if<std::is_convertible<T, DataT>::value &&              \
                              (std::is_fundamental<T>::value ||                \
                               std::is_same<T, half>::value),                  \
                          vec<rel_t, NumElements>>::type                       \
  operator RELLOGOP(const T &Rhs) const {                                      \
    return *this RELLOGOP vec(static_cast<const DataT &>(Rhs));                \
  }
#endif

  __SYCL_RELLOGOP(==)
  __SYCL_RELLOGOP(!=)
  __SYCL_RELLOGOP(>)
  __SYCL_RELLOGOP(<)
  __SYCL_RELLOGOP(>=)
  __SYCL_RELLOGOP(<=)
  // TODO: limit to integral types.
  __SYCL_RELLOGOP(&&)
  __SYCL_RELLOGOP(||)
#undef __SYCL_RELLOGOP

#ifdef __SYCL_UOP
#error "Undefine __SYCL_UOP macro"
#endif
#define __SYCL_UOP(UOP, OPASSIGN)                                              \
  vec &operator UOP() {                                                        \
    *this OPASSIGN 1;                                                          \
    return *this;                                                              \
  }                                                                            \
  vec operator UOP(int) {                                                      \
    vec Ret(*this);                                                            \
    *this OPASSIGN 1;                                                          \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  // Available only when: dataT != cl_float && dataT != cl_double
  // && dataT != cl_half
  template <typename T = DataT>
  typename std::enable_if<std::is_integral<T>::value, vec>::type
  operator~() const {
// Use __SYCL_DEVICE_ONLY__ macro because cast to OpenCL vector type is defined
// by SYCL device compiler only.
#ifdef __SYCL_DEVICE_ONLY__
    return vec{
      (typename vec::DataType)~m_Data};
#else
    vec Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret.setValue(I, ~getValue(I));
    }
    return Ret;
#endif
  }

  vec<rel_t, NumElements> operator!() const {
// Use __SYCL_DEVICE_ONLY__ macro because cast to OpenCL vector type is defined
// by SYCL device compiler only.
#ifdef __SYCL_DEVICE_ONLY__
    return vec<rel_t, NumElements>{
      (typename vec<rel_t, NumElements>::DataType)!m_Data};
#else
    vec<rel_t, NumElements> Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret.setValue(I, !getValue(I));
    }
    return Ret;
#endif
  }

  vec operator+() const {
// Use __SYCL_DEVICE_ONLY__ macro because cast to OpenCL vector type is defined
// by SYCL device compiler only.
#ifdef __SYCL_DEVICE_ONLY__
    return vec{+m_Data};
#else
    vec Ret;
    for (size_t I = 0; I < NumElements; ++I)
      Ret.setValue(I, +getValue(I));
    return Ret;
#endif
  }

  vec operator-() const {
// Use __SYCL_DEVICE_ONLY__ macro because cast to OpenCL vector type is defined
// by SYCL device compiler only.
#ifdef __SYCL_DEVICE_ONLY__
    return vec{-m_Data};
#else
    vec Ret;
    for (size_t I = 0; I < NumElements; ++I)
      Ret.setValue(I, -getValue(I));
    return Ret;
#endif
  }

  // OP is: &&, ||
  // vec<RET, NumElements> operatorOP(const vec<DataT, NumElements> &Rhs) const;
  // vec<RET, NumElements> operatorOP(const DataT &Rhs) const;

  // OP is: ==, !=, <, >, <=, >=
  // vec<RET, NumElements> operatorOP(const vec<DataT, NumElements> &Rhs) const;
  // vec<RET, NumElements> operatorOP(const DataT &Rhs) const;
private:
  // Generic method that execute "Operation" on underlying values.
#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
  template <template <typename> class Operation,
            typename Ty = vec<DataT, NumElements>>
  vec<DataT, NumElements>
  operatorHelper(const EnableIfNotHostHalf<Ty> &Rhs) const {
    vec<DataT, NumElements> Result;
    Operation<DataType> Op;
    Result.m_Data = Op(m_Data, Rhs.m_Data);
    return Result;
  }

  template <template <typename> class Operation,
            typename Ty = vec<DataT, NumElements>>
  vec<DataT, NumElements>
  operatorHelper(const EnableIfHostHalf<Ty> &Rhs) const {
    vec<DataT, NumElements> Result;
    Operation<DataT> Op;
    for (size_t I = 0; I < NumElements; ++I) {
      Result.setValue(I, Op(Rhs.getValue(I), getValue(I)));
    }
    return Result;
  }
#else  // __SYCL_USE_EXT_VECTOR_TYPE__
  template <template <typename> class Operation>
  vec<DataT, NumElements>
  operatorHelper(const vec<DataT, NumElements> &Rhs) const {
    vec<DataT, NumElements> Result;
    Operation<DataT> Op;
    for (size_t I = 0; I < NumElements; ++I) {
      Result.setValue(I, Op(Rhs.getValue(I), getValue(I)));
    }
    return Result;
  }
#endif // __SYCL_USE_EXT_VECTOR_TYPE__

// setValue and getValue should be able to operate on different underlying
// types: enum cl_float#N , builtin vector float#N, builtin type float.
#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
  template <int Num = NumElements, typename Ty = int,
            typename = typename std::enable_if<1 != Num>::type>
  void setValue(EnableIfNotHostHalf<Ty> Index, const DataT &Value, int) {
    m_Data[Index] = Value;
  }

  template <int Num = NumElements, typename Ty = int,
            typename = typename std::enable_if<1 != Num>::type>
  DataT getValue(EnableIfNotHostHalf<Ty> Index, int) const {
    return m_Data[Index];
  }

  template <int Num = NumElements, typename Ty = int,
            typename = typename std::enable_if<1 != Num>::type>
  void setValue(EnableIfHostHalf<Ty> Index, const DataT &Value, int) {
    m_Data.s[Index] = Value;
  }

  template <int Num = NumElements, typename Ty = int,
            typename = typename std::enable_if<1 != Num>::type>
  DataT getValue(EnableIfHostHalf<Ty> Index, int) const {
    return m_Data.s[Index];
  }
#else  // __SYCL_USE_EXT_VECTOR_TYPE__
  template <int Num = NumElements,
            typename = typename std::enable_if<1 != Num>::type>
  void setValue(int Index, const DataT &Value, int) {
    m_Data.s[Index] = Value;
  }

  template <int Num = NumElements,
            typename = typename std::enable_if<1 != Num>::type>
  DataT getValue(int Index, int) const {
    return m_Data.s[Index];
  }
#endif // __SYCL_USE_EXT_VECTOR_TYPE__

  template <int Num = NumElements,
            typename = typename std::enable_if<1 == Num>::type>
  void setValue(int Index, const DataT &Value, float) {
    m_Data = Value;
  }

  template <int Num = NumElements,
            typename = typename std::enable_if<1 == Num>::type>
  DataT getValue(int Index, float) const {
    return m_Data;
  }

  // Special proxies as specialization is not allowed in class scope.
  void setValue(int Index, const DataT &Value) {
    if (NumElements == 1)
      setValue(Index, Value, 0);
    else
      setValue(Index, Value, 0.f);
  }

  DataT getValue(int Index) const {
    return (NumElements == 1) ? getValue(Index, 0) : getValue(Index, 0.f);
  }

  // Helpers for variadic template constructor of vec.
  template <typename T, typename... argTN>
  int vaargCtorHelper(int Idx, const T &arg) {
    setValue(Idx, arg);
    return Idx + 1;
  }

  template <typename DataT_, int NumElements_>
  int vaargCtorHelper(int Idx, const vec<DataT_, NumElements_> &arg) {
    for (size_t I = 0; I < NumElements_; ++I) {
      setValue(Idx + I, arg.getValue(I));
    }
    return Idx + NumElements_;
  }

  template <typename DataT_, int NumElements_, typename T2, typename T3,
            template <typename> class T4, int... T5>
  int vaargCtorHelper(int Idx,
                      const detail::SwizzleOp<vec<DataT_, NumElements_>, T2, T3,
                                              T4, T5...> &arg) {
    size_t NumElems = sizeof...(T5);
    for (size_t I = 0; I < NumElems; ++I) {
      setValue(Idx + I, arg.getValue(I));
    }
    return Idx + NumElems;
  }

  template <typename DataT_, int NumElements_, typename T2, typename T3,
            template <typename> class T4, int... T5>
  int vaargCtorHelper(int Idx,
                      const detail::SwizzleOp<const vec<DataT_, NumElements_>,
                                              T2, T3, T4, T5...> &arg) {
    size_t NumElems = sizeof...(T5);
    for (size_t I = 0; I < NumElems; ++I) {
      setValue(Idx + I, arg.getValue(I));
    }
    return Idx + NumElems;
  }

  template <typename T1, typename... argTN>
  void vaargCtorHelper(int Idx, const T1 &arg, const argTN &... args) {
    int NewIdx = vaargCtorHelper(Idx, arg);
    vaargCtorHelper(NewIdx, args...);
  }

  template <typename DataT_, int NumElements_, typename... argTN>
  void vaargCtorHelper(int Idx, const vec<DataT_, NumElements_> &arg,
                       const argTN &... args) {
    int NewIdx = vaargCtorHelper(Idx, arg);
    vaargCtorHelper(NewIdx, args...);
  }

  // fields
  // Used "SYCL_ALIGNAS" instead "alignas" to handle MSVC compiler.
  // For MSVC compiler max alignment is 64, e.g. vec<double, 16> required
  // alignment of 128 and MSVC compiler cann't align a parameter with requested
  // alignment of 128.
  SYCL_ALIGNAS((detail::VectorAlignment<DataT, NumElements>::value))
      DataType m_Data;

  // friends
  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class detail::SwizzleOp;
  template <typename T1, int T2> friend class vec;
};

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = detail::enable_if_t<(std::is_same<T, U>::value && ...)>>
vec(T, U...)->vec<T, sizeof...(U) + 1>;
#endif

namespace detail {

// SwizzleOP represents expression templates that operate on vec.
// Actual computation performed on conversion or assignment operators.
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp {
  using DataT = typename VecT::element_type;
  using CommonDataT =
      typename std::common_type<typename OperationLeftT::DataT,
                                typename OperationRightT::DataT>::type;
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
  using EnableIfOneIndex = typename std::enable_if<
      1 == IdxNum && SwizzleOp::getNumElements() == IdxNum, T>::type;

  template <int IdxNum, typename T = void>
  using EnableIfMultipleIndexes = typename std::enable_if<
      1 != IdxNum && SwizzleOp::getNumElements() == IdxNum, T>::type;

  template <typename T>
  using EnableIfScalarType = typename std::enable_if<
      std::is_convertible<DataT, T>::value &&
      (std::is_fundamental<T>::value ||
       std::is_same<typename std::remove_const<T>::type, half>::value)>::type;

  template <typename T>
  using EnableIfNoScalarType = typename std::enable_if<
      !std::is_convertible<DataT, T>::value ||
      !(std::is_fundamental<T>::value ||
        std::is_same<typename std::remove_const<T>::type, half>::value)>::type;

  template <int... Indices>
  using Swizzle =
      SwizzleOp<VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

  template <int... Indices>
  using ConstSwizzle =
      SwizzleOp<const VecT, GetOp<DataT>, GetOp<DataT>, GetOp, Indices...>;

public:
  size_t get_count() const { return getNumElements(); }
  template <int Num = getNumElements()> size_t get_size() const {
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
  SwizzleOp &operator OPASSIGN(const DataT &Rhs) {                             \
    operatorHelper<OP>(vec_t(Rhs));                                            \
    return *this;                                                              \
  }                                                                            \
  template <typename RhsOperation>                                             \
  SwizzleOp &operator OPASSIGN(const RhsOperation &Rhs) {                      \
    operatorHelper<OP>(Rhs);                                                   \
    return *this;                                                              \
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
  SwizzleOp &operator UOP() {                                                  \
    *this OPASSIGN static_cast<DataT>(1);                                      \
    return *this;                                                              \
  }                                                                            \
  vec_t operator UOP(int) {                                                    \
    vec_t Ret = *this;                                                         \
    *this OPASSIGN static_cast<DataT>(1);                                      \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  template <typename T = DataT>
  typename std::enable_if<std::is_integral<T>::value, vec_t>::type
  operator~() {
    vec_t Tmp = *this;
    return ~Tmp;
  }

  vec_rel_t operator!() {
    vec_t Tmp = *this;
    return !Tmp;
  }

  vec_t operator+() {
    vec_t Tmp = *this;
    return +Tmp;
  }

  vec_t operator-() {
    vec_t Tmp = *this;
    return -Tmp;
  }

  template <int IdxNum = getNumElements(),
            typename = EnableIfMultipleIndexes<IdxNum>>
  SwizzleOp &operator=(const vec<DataT, IdxNum> &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      m_Vector->setValue(Idxs[I], Rhs.getValue(I));
    }
    return *this;
  }

  template <int IdxNum = getNumElements(), typename = EnableIfOneIndex<IdxNum>>
  SwizzleOp &operator=(const DataT &Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    m_Vector->setValue(Idxs[0], Rhs);
    return *this;
  }

  template <int IdxNum = getNumElements(), typename = EnableIfOneIndex<IdxNum>>
  SwizzleOp &operator=(DataT &&Rhs) {
    std::array<int, IdxNum> Idxs{Indexes...};
    m_Vector->setValue(Idxs[0], Rhs);
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
            typename = typename std::enable_if<sizeof...(T5) ==
                                               getNumElements()>::type>
  SwizzleOp &operator=(const SwizzleOp<T1, T2, T3, T4, T5...> &Rhs) {
    std::array<int, getNumElements()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      m_Vector->setValue(Idxs[I], Rhs.getValue(I));
    }
    return *this;
  }

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5,
            typename = typename std::enable_if<sizeof...(T5) ==
                                               getNumElements()>::type>
  SwizzleOp &operator=(SwizzleOp<T1, T2, T3, T4, T5...> &&Rhs) {
    std::array<int, getNumElements()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      m_Vector->setValue(Idxs[I], Rhs.getValue(I));
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
  template <int Index>
  struct Indexer {
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

  // TODO: make templated address space to work.
  // Somehow, access<> to multi_ptr<> conversion doesn't work w/o making
  // address space explicitly specified.
  //
  // Leave store() interface to automatic conversion to vec<>.
  // Load to vec_t and then assign to swizzle.
#ifdef __SYCL_LOAD
#error "Undefine __SYCL_LOAD macro"
#endif
#define __SYCL_LOAD(Space)                                                     \
  void load(size_t offset, multi_ptr<DataT, Space> ptr) {                      \
    vec_t Tmp;                                                                 \
    Tmp.template load(offset, ptr);                                            \
    *this = Tmp;                                                               \
  }

  __SYCL_LOAD(access::address_space::global_space)
  __SYCL_LOAD(access::address_space::local_space)
  __SYCL_LOAD(access::address_space::constant_space)
  __SYCL_LOAD(access::address_space::private_space)
#undef __SYCL_LOAD

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, sizeof...(Indexes)> convert() const {
    // First materialize the swizzle to vec_t and then apply convert() to it.
    vec_t Tmp = *this;
    return Tmp.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  typename std::enable_if<asT::getNumElements() == getNumElements(), asT>::type
  as() const {
    // First materialize the swizzle to vec_t and then apply as() to it.
    vec_t Tmp = *this;
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
    if (std::is_same<OperationCurrentT<DataT>, GetOp<DataT>>::value) {
      std::array<int, getNumElements()> Idxs{Indexes...};
      return m_Vector->getValue(Idxs[Index]);
    }
    auto Op = OperationCurrentT<CommonDataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <int IdxNum = getNumElements()>
  DataT getValue(EnableIfMultipleIndexes<IdxNum, size_t> Index) const {
    if (std::is_same<OperationCurrentT<DataT>, GetOp<DataT>>::value) {
      std::array<int, getNumElements()> Idxs{Indexes...};
      return m_Vector->getValue(Idxs[Index]);
    }
    auto Op = OperationCurrentT<DataT>();
    return Op(m_LeftOperation.getValue(Index),
              m_RightOperation.getValue(Index));
  }

  template <template <typename> class Operation, typename RhsOperation>
  void operatorHelper(const RhsOperation &Rhs) {
    Operation<DataT> Op;
    std::array<int, getNumElements()> Idxs{Indexes...};
    for (size_t I = 0; I < Idxs.size(); ++I) {
      DataT Res = Op(m_Vector->getValue(Idxs[I]), Rhs.getValue(I));
      m_Vector->setValue(Idxs[I], Res);
    }
  }

  // fields
  VecT *m_Vector;

  OperationLeftT m_LeftOperation;
  OperationRightT m_RightOperation;

  // friends
  template <typename T1, int T2> friend class cl::sycl::vec;

  template <typename T1, typename T2, typename T3, template <typename> class T4,
            int... T5>
  friend class SwizzleOp;
};
} // namespace detail

// scalar BINOP vec<>
// scalar BINOP SwizzleOp
// vec<> BINOP SwizzleOp
#ifdef __SYCL_BINOP
#error "Undefine __SYCL_BINOP macro"
#endif
#define __SYCL_BINOP(BINOP)                                                    \
  template <typename T, int Num>                                               \
  typename std::enable_if<                                                     \
      std::is_fundamental<T>::value ||                                         \
          std::is_same<typename std::remove_const<T>::type, half>::value,      \
      vec<T, Num>>::type                                                       \
  operator BINOP(const T &Lhs, const vec<T, Num> &Rhs) {                       \
    return vec<T, Num>(Lhs) BINOP Rhs;                                         \
  }                                                                            \
  template <typename VecT, typename OperationLeftT, typename OperationRightT,  \
            template <typename> class OperationCurrentT, int... Indexes,       \
            typename T, typename T1 = typename VecT::element_type,             \
            int Num = sizeof...(Indexes)>                                      \
  typename std::enable_if<                                                     \
      std::is_convertible<T, T1>::value &&                                     \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      vec<T1, Num>>::type                                                      \
  operator BINOP(const T &Lhs,                                                 \
                 const detail::SwizzleOp<VecT, OperationLeftT, OperationRightT,\
                                         OperationCurrentT, Indexes...> &Rhs) {\
    vec<T1, Num> Tmp = Rhs;                                                    \
    return Lhs BINOP Tmp;                                                      \
  }                                                                            \
  template <typename VecT, typename OperationLeftT, typename OperationRightT,  \
            template <typename> class OperationCurrentT, int... Indexes,       \
            typename T = typename VecT::element_type,                          \
            int Num = sizeof...(Indexes)>                                      \
  vec<T, Num> operator BINOP(                                                  \
      const vec<T, Num> &Lhs,                                                  \
      const detail::SwizzleOp<VecT, OperationLeftT, OperationRightT,           \
                              OperationCurrentT, Indexes...> &Rhs) {           \
    vec<T, Num> Tmp = Rhs;                                                     \
    return Lhs BINOP Tmp;                                                      \
  }

__SYCL_BINOP(+)
__SYCL_BINOP(-)
__SYCL_BINOP(*)
__SYCL_BINOP(/)
__SYCL_BINOP(&)
__SYCL_BINOP(|)
__SYCL_BINOP(^)
__SYCL_BINOP(>>)
__SYCL_BINOP(<<)
#undef __SYCL_BINOP

// scalar RELLOGOP vec<>
// scalar RELLOGOP SwizzleOp
// vec<> RELLOGOP SwizzleOp
#ifdef __SYCL_RELLOGOP
#error "Undefine __SYCL_RELLOGOP macro"
#endif
#define __SYCL_RELLOGOP(RELLOGOP)                                              \
  template <typename T, typename DataT, int Num>                               \
  typename std::enable_if<                                                     \
      std::is_convertible<T, DataT>::value &&                                  \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      vec<detail::rel_t<DataT>, Num>>::type                                    \
  operator RELLOGOP(const T &Lhs, const vec<DataT, Num> &Rhs) {                \
    return vec<T, Num>(static_cast<T>(Lhs)) RELLOGOP Rhs;                      \
  }                                                                            \
  template <typename VecT, typename OperationLeftT, typename OperationRightT,  \
            template <typename> class OperationCurrentT, int... Indexes,       \
            typename T, typename T1 = typename VecT::element_type,             \
            int Num = sizeof...(Indexes)>                                      \
  typename std::enable_if<                                                     \
      std::is_convertible<T, T1>::value &&                                     \
          (std::is_fundamental<T>::value ||                                    \
           std::is_same<typename std::remove_const<T>::type, half>::value),    \
      vec<detail::rel_t<T1>, Num>>::type                                       \
  operator RELLOGOP(                                                           \
      const T &Lhs,                                                            \
      const detail::SwizzleOp<VecT, OperationLeftT, OperationRightT,           \
                              OperationCurrentT, Indexes...> &Rhs) {           \
    vec<T1, Num> Tmp = Rhs;                                                    \
    return Lhs RELLOGOP Tmp;                                                   \
  }                                                                            \
  template <typename VecT, typename OperationLeftT, typename OperationRightT,  \
            template <typename> class OperationCurrentT, int... Indexes,       \
            typename T = typename VecT::element_type,                          \
            int Num = sizeof...(Indexes)>                                      \
  vec<detail::rel_t<T>, Num> operator RELLOGOP(                                \
      const vec<T, Num> &Lhs,                                                  \
      const detail::SwizzleOp<VecT, OperationLeftT, OperationRightT,           \
                              OperationCurrentT, Indexes...> &Rhs) {           \
    vec<T, Num> Tmp = Rhs;                                                     \
    return Lhs RELLOGOP Tmp;                                                   \
  }

__SYCL_RELLOGOP(==)
__SYCL_RELLOGOP(!=)
__SYCL_RELLOGOP(>)
__SYCL_RELLOGOP(<)
__SYCL_RELLOGOP(>=)
__SYCL_RELLOGOP(<=)
// TODO: limit to integral types.
__SYCL_RELLOGOP(&&)
__SYCL_RELLOGOP(||)
#undef __SYCL_RELLOGOP

} // namespace sycl
} // namespace cl


#ifdef __SYCL_USE_EXT_VECTOR_TYPE__
#define DECLARE_TYPE_VIA_CL_T(type)                                            \
using __##type##_t       = cl::sycl::cl_##type;        \
using __##type##2_vec_t  = cl::sycl::cl_##type         \
                 __attribute__((ext_vector_type(2)));  \
using __##type##3_vec_t  = cl::sycl::cl_##type         \
                 __attribute__((ext_vector_type(3)));  \
using __##type##4_vec_t  = cl::sycl::cl_##type         \
                 __attribute__((ext_vector_type(4)));  \
using __##type##8_vec_t  = cl::sycl::cl_##type         \
                 __attribute__((ext_vector_type(8)));  \
using __##type##16_vec_t = cl::sycl::cl_##type         \
                 __attribute__((ext_vector_type(16)));

#define DECLARE_TYPE_T(type)                                                   \
using __##type##_t       = cl::sycl::type;             \
using __##type##2_vec_t  = cl::sycl::type              \
                 __attribute__((ext_vector_type(2)));  \
using __##type##3_vec_t  = cl::sycl::type              \
                 __attribute__((ext_vector_type(3)));  \
using __##type##4_vec_t  = cl::sycl::type              \
                 __attribute__((ext_vector_type(4)));  \
using __##type##8_vec_t  = cl::sycl::type              \
                 __attribute__((ext_vector_type(8)));  \
using __##type##16_vec_t = cl::sycl::type              \
                 __attribute__((ext_vector_type(16)));

DECLARE_TYPE_VIA_CL_T(char);
DECLARE_TYPE_T(schar);
DECLARE_TYPE_VIA_CL_T(uchar);
DECLARE_TYPE_VIA_CL_T(short);
DECLARE_TYPE_VIA_CL_T(ushort);
DECLARE_TYPE_VIA_CL_T(int);
DECLARE_TYPE_VIA_CL_T(uint);
DECLARE_TYPE_VIA_CL_T(long);
DECLARE_TYPE_VIA_CL_T(ulong);
DECLARE_TYPE_T(longlong);
DECLARE_TYPE_T(ulonglong);
DECLARE_TYPE_VIA_CL_T(float);
// Half type is defined as custom class for host and _Float16 for device.
// The ext_vector_type attribute is only applicable to integral and float
// scalars so it's not possible to use attribute ext_vector_type for half on
// host.
#ifdef __SYCL_DEVICE_ONLY__
DECLARE_TYPE_VIA_CL_T(half);
#endif
DECLARE_TYPE_VIA_CL_T(double);

#define GET_CL_TYPE(target, num) __##target##num##_vec_t
#define GET_SCALAR_CL_TYPE(target) target

#undef DECLARE_TYPE_VIA_CL_T
#undef DECLARE_TYPE_T
#else // __SYCL_USE_EXT_VECTOR_TYPE__
#define GET_CL_TYPE(target, num) ::cl_##target##num
#define GET_SCALAR_CL_TYPE(target) ::cl_##target
#endif // __SYCL_USE_EXT_VECTOR_TYPE__

// On the host side we cannot use OpenCL cl_half# types as an underlying type
// for vec because they are actually defined as an integer type under the hood.
// As a result half values will be converted to the integer and passed as a
// kernel argument which is expected to be floating point number.
#ifndef __SYCL_DEVICE_ONLY__
template <int NumElements> struct half_vec {
  alignas(cl::sycl::detail::VectorAlignment<half, NumElements>::value)
      std::array<half, NumElements> s;
};

using __half_t = half;
using __half2_vec_t = half_vec<2>;
using __half3_vec_t = half_vec<3>;
using __half4_vec_t = half_vec<4>;
using __half8_vec_t = half_vec<8>;
using __half16_vec_t = half_vec<16>;
#endif

#define GET_CL_HALF_TYPE(target, num) __##target##num##_vec_t

namespace cl {
namespace sycl {
namespace detail {
// select_apply_cl_t selects from T8/T16/T32/T64 basing on
// sizeof(IN).  expected to handle scalar types in IN.
template <typename T, typename T8, typename T16, typename T32, typename T64>
using select_apply_cl_t =
  conditional_t<sizeof(T) == 1, T8,
  conditional_t<sizeof(T) == 2, T16,
  conditional_t<sizeof(T) == 4, T32, T64>>>;
} // detail

#define DECLARE_CONVERTER(base, num)                                           \
  template <> class BaseCLTypeConverter<base, num> {                           \
  public:                                                                      \
    using DataType = GET_CL_TYPE(base, num);                                   \
  };

#define DECLARE_SIGNED_INTEGRAL_CONVERTER(base, num)                           \
  template <> class BaseCLTypeConverter<base, num> {                           \
  public:                                                                      \
    using DataType = detail::select_apply_cl_t<base, GET_CL_TYPE(char, num),   \
      GET_CL_TYPE(short, num), GET_CL_TYPE(int, num), GET_CL_TYPE(long, num)>; \
  };

#define DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, num)                         \
  template <> class BaseCLTypeConverter<base, num> {                           \
  public:                                                                      \
    using DataType = detail::select_apply_cl_t<base, GET_CL_TYPE(uchar, num),  \
      GET_CL_TYPE(ushort, num), GET_CL_TYPE(uint, num),                        \
      GET_CL_TYPE(ulong, num)>;                                                \
  };

#define DECLARE_FLOAT_CONVERTER(base, num)                                     \
  template <> class BaseCLTypeConverter<base, num> {                           \
  public:                                                                      \
    using DataType = detail::select_apply_cl_t<base, std::false_type,          \
      GET_CL_HALF_TYPE(half, num), GET_CL_TYPE(float, num),                    \
      GET_CL_TYPE(double, num)>;                                               \
  };

#define DECLARE_LONGLONG_CONVERTER(base, num)                                  \
  template <> class BaseCLTypeConverter<base##long, num> {                     \
  public:                                                                      \
    using DataType = GET_CL_TYPE(base, num);                                   \
  };

#define DECLARE_SCHAR_CONVERTER(num)                                           \
  template <> class BaseCLTypeConverter<schar, num> {                          \
  public:                                                                      \
    using DataType = detail::select_apply_cl_t<                                \
        schar, GET_CL_TYPE(char, num), GET_CL_TYPE(short, num),                \
        GET_CL_TYPE(int, num), GET_CL_TYPE(long, num)>;                        \
  };

#define DECLARE_HALF_CONVERTER(base, num)                                      \
  template <> class BaseCLTypeConverter<base, num> {                           \
  public:                                                                      \
    using DataType = GET_CL_HALF_TYPE(base, num);                              \
  };

#define DECLARE_SCALAR_SCHAR_CONVERTER                                         \
  template <> class BaseCLTypeConverter<schar, 1> {                            \
  public:                                                                      \
    using DataType = schar;                                                    \
  };

#define DECLARE_SCALAR_CONVERTER(base)                                         \
  template <> class BaseCLTypeConverter<base, 1> {                             \
  public:                                                                      \
    using DataType = GET_SCALAR_CL_TYPE(base);                                 \
  };

#define DECLARE_VECTOR_CONVERTERS(base)                                        \
  namespace detail {                                                           \
  DECLARE_CONVERTER(base, 2)                                                   \
  DECLARE_CONVERTER(base, 3)                                                   \
  DECLARE_CONVERTER(base, 4)                                                   \
  DECLARE_CONVERTER(base, 8)                                                   \
  DECLARE_CONVERTER(base, 16)                                                  \
  DECLARE_SCALAR_CONVERTER(base)                                               \
  } // namespace detail

#define DECLARE_SIGNED_INTEGRAL_VECTOR_CONVERTERS(base)                        \
  namespace detail {                                                           \
  DECLARE_SIGNED_INTEGRAL_CONVERTER(base, 2)                                   \
  DECLARE_SIGNED_INTEGRAL_CONVERTER(base, 3)                                   \
  DECLARE_SIGNED_INTEGRAL_CONVERTER(base, 4)                                   \
  DECLARE_SIGNED_INTEGRAL_CONVERTER(base, 8)                                   \
  DECLARE_SIGNED_INTEGRAL_CONVERTER(base, 16)                                  \
  DECLARE_SCALAR_CONVERTER(base)                                               \
  } // namespace detail

#define DECLARE_UNSIGNED_INTEGRAL_VECTOR_CONVERTERS(base)                      \
  namespace detail {                                                           \
  DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, 2)                                 \
  DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, 3)                                 \
  DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, 4)                                 \
  DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, 8)                                 \
  DECLARE_UNSIGNED_INTEGRAL_CONVERTER(base, 16)                                \
  DECLARE_SCALAR_CONVERTER(base)                                               \
  } // namespace detail

#define DECLARE_FLOAT_VECTOR_CONVERTERS(base)                                  \
  namespace detail {                                                           \
  DECLARE_FLOAT_CONVERTER(base, 2)                                             \
  DECLARE_FLOAT_CONVERTER(base, 3)                                             \
  DECLARE_FLOAT_CONVERTER(base, 4)                                             \
  DECLARE_FLOAT_CONVERTER(base, 8)                                             \
  DECLARE_FLOAT_CONVERTER(base, 16)                                            \
  DECLARE_SCALAR_CONVERTER(base)                                               \
  } // namespace detail


#define DECLARE_HALF_VECTOR_CONVERTERS(base)                                   \
  namespace detail {                                                           \
  DECLARE_HALF_CONVERTER(base, 2)                                              \
  DECLARE_HALF_CONVERTER(base, 3)                                              \
  DECLARE_HALF_CONVERTER(base, 4)                                              \
  DECLARE_HALF_CONVERTER(base, 8)                                              \
  DECLARE_HALF_CONVERTER(base, 16)                                             \
  template <> class BaseCLTypeConverter<base, 1> {                             \
  public:                                                                      \
    using DataType = half;                                                     \
  };                                                                           \
  } // namespace detail

#define DECLARE_VECTOR_LONGLONG_CONVERTERS(base)                               \
  namespace detail {                                                           \
  DECLARE_LONGLONG_CONVERTER(base, 2)                                          \
  DECLARE_LONGLONG_CONVERTER(base, 3)                                          \
  DECLARE_LONGLONG_CONVERTER(base, 4)                                          \
  DECLARE_LONGLONG_CONVERTER(base, 8)                                          \
  DECLARE_LONGLONG_CONVERTER(base, 16)                                         \
  template <> class BaseCLTypeConverter<base##long, 1> {                       \
  public:                                                                      \
    using DataType = base##long;                                               \
  };                                                                           \
  } // namespace detail

#define DECLARE_SCHAR_VECTOR_CONVERTERS                                        \
  namespace detail {                                                           \
  DECLARE_SCHAR_CONVERTER(2)                                                   \
  DECLARE_SCHAR_CONVERTER(3)                                                   \
  DECLARE_SCHAR_CONVERTER(4)                                                   \
  DECLARE_SCHAR_CONVERTER(8)                                                   \
  DECLARE_SCHAR_CONVERTER(16)                                                  \
  DECLARE_SCALAR_SCHAR_CONVERTER                                               \
  } // namespace detail

DECLARE_VECTOR_CONVERTERS(char)
DECLARE_SCHAR_VECTOR_CONVERTERS
DECLARE_UNSIGNED_INTEGRAL_VECTOR_CONVERTERS(uchar)
DECLARE_SIGNED_INTEGRAL_VECTOR_CONVERTERS(short)
DECLARE_UNSIGNED_INTEGRAL_VECTOR_CONVERTERS(ushort)
DECLARE_SIGNED_INTEGRAL_VECTOR_CONVERTERS(int)
DECLARE_UNSIGNED_INTEGRAL_VECTOR_CONVERTERS(uint)
DECLARE_SIGNED_INTEGRAL_VECTOR_CONVERTERS(long)
DECLARE_UNSIGNED_INTEGRAL_VECTOR_CONVERTERS(ulong)
DECLARE_VECTOR_LONGLONG_CONVERTERS(long)
DECLARE_VECTOR_LONGLONG_CONVERTERS(ulong)
DECLARE_HALF_VECTOR_CONVERTERS(half)
DECLARE_FLOAT_VECTOR_CONVERTERS(float)
DECLARE_FLOAT_VECTOR_CONVERTERS(double)

#undef GET_CL_TYPE
#undef GET_SCALAR_CL_TYPE
#undef DECLARE_CONVERTER
#undef DECLARE_VECTOR_CONVERTERS
#undef DECLARE_SYCL_VEC
#undef DECLARE_SYCL_VEC_WO_CONVERTERS
#undef DECLARE_SCHAR_VECTOR_CONVERTERS
#undef DECLARE_SCHAR_CONVERTER
#undef DECLARE_SCALAR_SCHAR_CONVERTER

} // namespace sycl
} // namespace cl

#undef SYCL_ALIGNAS
