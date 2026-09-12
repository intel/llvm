//=== vector_arith.hpp --- Implementation of arithmetic ops on sycl::vec  ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/type_traits/vec_marray_traits.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T> struct from_incomplete;
template <typename T>
struct from_incomplete<const T> : public from_incomplete<T> {};

template <typename DataT, int NumElements>
struct from_incomplete<vec<DataT, NumElements>> {
  using element_type = DataT;
  static constexpr size_t size() { return NumElements; }
};

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct from_incomplete<
    hide_swizzle_from_adl::Swizzle<IsConstVec, DataT, VecSize, Indexes...>> {
  using element_type = DataT;
  static constexpr size_t size() { return sizeof...(Indexes); }

  using vec_ty = std::conditional_t<IsConstVec, const vec<DataT, VecSize>,
                                    vec<DataT, VecSize>>;
  using result_vec_ty = vec<DataT, size()>;
  static constexpr int vec_size = VecSize;
  static constexpr bool is_over_const_vec = IsConstVec;
  static constexpr bool has_repeating_indexes = []() constexpr {
    int Idxs[] = {Indexes...};
    for (std::size_t i = 1; i < sizeof...(Indexes); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (Idxs[j] == Idxs[i])
          // Repeating index
          return true;
    }

    return false;
  }();
  static constexpr bool is_assignable = !IsConstVec && !has_repeating_indexes;
};
#endif

template <bool Cond, typename Mixin> struct ApplyIf {};
template <typename Mixin> struct ApplyIf<true, Mixin> : Mixin {};

// We use std::plus<void> and similar to "map" template parameter to an
// overloaded operator. These three below are missing from `<functional>`.
struct ShiftLeft {
  template <class T, class U>
  constexpr auto operator()(T &&lhs, U &&rhs) const
      -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
};
struct ShiftRight {
  template <class T, class U>
  constexpr auto operator()(T &&lhs,
                            U &&rhs) const -> decltype(std::forward<T>(lhs) >>
                                                       std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

struct UnaryPlus {
  template <class T>
  constexpr auto operator()(T &&arg) const -> decltype(+std::forward<T>(arg)) {
    return +std::forward<T>(arg);
  }
};

template <typename Op> struct OpAssign {};

// Tag to map/templatize the mixin for prefix/postfix inc/dec operators.
struct IncDec {};

template <class T> static constexpr bool not_fp = !is_vgenfloat_v<T>;

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
// Not using `is_byte_v` to avoid unnecessary dependencies on `half`/`bfloat16`
// headers.
template <class T>
static constexpr bool not_byte =
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
    !std::is_same_v<T, std::byte>;
#else
    true;
#endif
#endif

// To provide information about operators availability depending on vec/swizzle
// element type.
template <typename Op, typename T>
inline constexpr bool is_op_available_for_type = false;

template <typename Op, typename T>
inline constexpr bool is_op_available_for_type<OpAssign<Op>, T> =
    is_op_available_for_type<Op, T>;

#define __SYCL_OP_AVAILABILITY(OP, COND)                                       \
  template <typename T>                                                        \
  inline constexpr bool is_op_available_for_type<OP, T> = COND;

// clang-format off
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
__SYCL_OP_AVAILABILITY(std::plus<void>          , true)
__SYCL_OP_AVAILABILITY(std::minus<void>         , true)
__SYCL_OP_AVAILABILITY(std::multiplies<void>    , true)
__SYCL_OP_AVAILABILITY(std::divides<void>       , true)
__SYCL_OP_AVAILABILITY(std::modulus<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::bit_and<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_or<void>        , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_xor<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::equal_to<void>      , true)
__SYCL_OP_AVAILABILITY(std::not_equal_to<void>  , true)
__SYCL_OP_AVAILABILITY(std::less<void>          , true)
__SYCL_OP_AVAILABILITY(std::greater<void>       , true)
__SYCL_OP_AVAILABILITY(std::less_equal<void>    , true)
__SYCL_OP_AVAILABILITY(std::greater_equal<void> , true)

__SYCL_OP_AVAILABILITY(std::logical_and<void>   , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::logical_or<void>    , not_fp<T>)

__SYCL_OP_AVAILABILITY(ShiftLeft                , not_fp<T>)
__SYCL_OP_AVAILABILITY(ShiftRight               , not_fp<T>)

// Unary
__SYCL_OP_AVAILABILITY(std::negate<void>        , true)
__SYCL_OP_AVAILABILITY(std::logical_not<void>   , true)
__SYCL_OP_AVAILABILITY(std::bit_not<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(UnaryPlus                , true)

__SYCL_OP_AVAILABILITY(IncDec                   , true)
#else
__SYCL_OP_AVAILABILITY(std::plus<void>          , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::minus<void>         , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::multiplies<void>    , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::divides<void>       , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::modulus<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::bit_and<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_or<void>        , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_xor<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::equal_to<void>      , true)
__SYCL_OP_AVAILABILITY(std::not_equal_to<void>  , true)
__SYCL_OP_AVAILABILITY(std::less<void>          , true)
__SYCL_OP_AVAILABILITY(std::greater<void>       , true)
__SYCL_OP_AVAILABILITY(std::less_equal<void>    , true)
__SYCL_OP_AVAILABILITY(std::greater_equal<void> , true)

__SYCL_OP_AVAILABILITY(std::logical_and<void>   , not_byte<T> && not_fp<T>)
__SYCL_OP_AVAILABILITY(std::logical_or<void>    , not_byte<T> && not_fp<T>)

__SYCL_OP_AVAILABILITY(ShiftLeft                , not_byte<T> && not_fp<T>)
__SYCL_OP_AVAILABILITY(ShiftRight               , not_byte<T> && not_fp<T>)

// Unary
__SYCL_OP_AVAILABILITY(std::negate<void>        , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::logical_not<void>   , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::bit_not<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(UnaryPlus                , not_byte<T>)

__SYCL_OP_AVAILABILITY(IncDec                   , not_byte<T>)
#endif
// clang-format on

#undef __SYCL_OP_AVAILABILITY

template <typename SelfOperandTy> struct IncDecImpl {
  using element_type = typename from_incomplete<SelfOperandTy>::element_type;
  using vec_t = simplify_if_swizzle_t<std::remove_const_t<SelfOperandTy>>;

public:
  friend SelfOperandTy &operator++(SelfOperandTy &x) {
    x += element_type{1};
    return x;
  }
  friend SelfOperandTy &operator--(SelfOperandTy &x) {
    x -= element_type{1};
    return x;
  }
  friend auto operator++(SelfOperandTy &x, int) {
    vec_t tmp{x};
    x += element_type{1};
    return tmp;
  }
  friend auto operator--(SelfOperandTy &x, int) {
    vec_t tmp{x};
    x -= element_type{1};
    return tmp;
  }
};

// clang-format off
#define __SYCL_INSTANTIATE_OPERATORS(BINOP, OPASSIGN, UOP)                     \
BINOP(std::plus<void>          , +)                                            \
BINOP(std::minus<void>         , -)                                            \
BINOP(std::multiplies<void>    , *)                                            \
BINOP(std::divides<void>       , /)                                            \
BINOP(std::modulus<void>       , %)                                            \
BINOP(std::bit_and<void>       , &)                                            \
BINOP(std::bit_or<void>        , |)                                            \
BINOP(std::bit_xor<void>       , ^)                                            \
BINOP(std::equal_to<void>      , ==)                                           \
BINOP(std::not_equal_to<void>  , !=)                                           \
BINOP(std::less<void>          , < )                                           \
BINOP(std::greater<void>       , >)                                            \
BINOP(std::less_equal<void>    , <=)                                           \
BINOP(std::greater_equal<void> , >=)                                           \
BINOP(std::logical_and<void>   , &&)                                           \
BINOP(std::logical_or<void>    , ||)                                           \
BINOP(ShiftLeft                , <<)                                           \
BINOP(ShiftRight               , >>)                                           \
UOP(std::negate<void>          , -)                                            \
UOP(std::logical_not<void>     , !)                                            \
UOP(std::bit_not<void>         , ~)                                            \
UOP(UnaryPlus                  , +)                                            \
OPASSIGN(std::plus<void>       , +=)                                           \
OPASSIGN(std::minus<void>      , -=)                                           \
OPASSIGN(std::multiplies<void> , *=)                                           \
OPASSIGN(std::divides<void>    , /=)                                           \
OPASSIGN(std::modulus<void>    , %=)                                           \
OPASSIGN(std::bit_and<void>    , &=)                                           \
OPASSIGN(std::bit_or<void>     , |=)                                           \
OPASSIGN(std::bit_xor<void>    , ^=)                                           \
OPASSIGN(ShiftLeft             , <<=)                                          \
OPASSIGN(ShiftRight            , >>=)
// clang-format on

template <typename Op>
constexpr bool is_logical =
    check_type_in_v<Op, std::equal_to<void>, std::not_equal_to<void>,
                    std::less<void>, std::greater<void>, std::less_equal<void>,
                    std::greater_equal<void>, std::logical_and<void>,
                    std::logical_or<void>, std::logical_not<void>>;

template <typename Self> struct VecOperators {
  static_assert(is_vec_v<Self>);

  using element_type = typename from_incomplete<Self>::element_type;
  static constexpr int N = from_incomplete<Self>::size();

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
  template <typename T>
  static constexpr bool is_compatible_scalar =
      std::is_convertible_v<T, typename from_incomplete<Self>::element_type>;
#endif

  template <typename Op>
  using result_t = std::conditional_t<
      is_logical<Op>, vec<fixed_width_signed<sizeof(element_type)>, N>, Self>;

  template <typename OpTy, typename... ArgTys>
  static constexpr auto apply(const ArgTys &...Args) {
    static_assert(((std::is_same_v<Self, ArgTys> && ...)));

    OpTy Op{};
#ifdef __has_extension
#if __has_extension(attribute_ext_vector_type)
    // ext_vector_type's bool vectors are mapped onto <N x i1> and have
    // different memory layout than sycl::vec<bool ,N> (which has 1 byte per
    // element). As such we perform operation on int8_t and then need to
    // create bit pattern that can be bit-casted back to the original
    // sycl::vec<bool, N>. This is a hack actually, but we've been doing
    // that for a long time using sycl::vec::vector_t type.
    using vec_elem_ty =
        typename detail::map_type<element_type, //
                                  bool, /*->*/ std::int8_t,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
                                  std::byte, /*->*/ std::uint8_t,
#endif
#ifdef __SYCL_DEVICE_ONLY__
                                  half, /*->*/ _Float16,
#endif
                                  element_type, /*->*/ element_type>::type;
    if constexpr (N != 1 &&
                  detail::is_valid_type_for_ext_vector_v<vec_elem_ty>) {
      using vec_t = ext_vector<vec_elem_ty, N>;
      auto tmp = [&](auto... xs) {
        // Workaround for https://github.com/llvm/llvm-project/issues/119617.
        if constexpr (sizeof...(Args) == 2) {
          return [&](auto x, auto y) {
            if constexpr (std::is_same_v<OpTy, std::equal_to<void>>)
              return x == y;
            else if constexpr (std::is_same_v<OpTy, std::not_equal_to<void>>)
              return x != y;
            else if constexpr (std::is_same_v<OpTy, std::less<void>>)
              return x < y;
            else if constexpr (std::is_same_v<OpTy, std::less_equal<void>>)
              return x <= y;
            else if constexpr (std::is_same_v<OpTy, std::greater<void>>)
              return x > y;
            else if constexpr (std::is_same_v<OpTy, std::greater_equal<void>>)
              return x >= y;
            else
              return Op(x, y);
          }(xs...);
        } else {
          return Op(xs...);
        }
      }(bit_cast<vec_t>(Args)...);
      if constexpr (std::is_same_v<element_type, bool>) {
        // Some operations are known to produce the required bit patterns and
        // the following post-processing isn't necessary for them:
        if constexpr (!is_logical<OpTy> &&
                      !check_type_in_v<OpTy, std::multiplies<void>,
                                       std::divides<void>, std::bit_or<void>,
                                       std::bit_and<void>, std::bit_xor<void>,
                                       ShiftRight, UnaryPlus>) {
          // Extra cast is needed because:
          static_assert(std::is_same_v<int8_t, signed char>);
          static_assert(!std::is_same_v<
                        decltype(std::declval<ext_vector<int8_t, 2>>() != 0),
                        ext_vector<int8_t, 2>>);
          static_assert(std::is_same_v<
                        decltype(std::declval<ext_vector<int8_t, 2>>() != 0),
                        ext_vector<char, 2>>);

          // `... * -1` is needed because ext_vector_type's comparison follows
          // OpenCL binary representation for "true" (-1).
          // `std::array<bool, N>` is different and LLVM annotates its
          // elements with [0, 2) range metadata when loaded, so we need to
          // ensure we generate 0/1 only (and not 2/-1/etc.).
#if __clang_major__ >= 20
          // Not an integral constant expression prior to clang-20.
          static_assert(
              static_cast<int8_t>((ext_vector<int8_t, 2>{1, 0} == 0)[1]) == -1);
#endif

          tmp = reinterpret_cast<decltype(tmp)>((tmp != 0) * -1);
        }
      }
      return bit_cast<result_t<OpTy>>(tmp);
    }
#endif
#endif
    result_t<OpTy> res{};
    for (size_t i = 0; i < N; ++i)
      if constexpr (is_logical<OpTy>)
        res[i] = Op(Args[i]...) ? -1 : 0;
      else
        res[i] = Op(Args[i]...);
    return res;
  }

  // Uglier than possible due to
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282.
  template <typename Op, typename = void> struct OpMixin;

  template <typename Op>
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, IncDec>>>
      : public IncDecImpl<Self> {};

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
#define __SYCL_VEC_BINOP_MIXIN(OP, OPERATOR)                                   \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OP>>> {               \
    template <typename T = element_type>                                       \
    friend std::enable_if_t<is_op_available_for_type<OP, T>, result_t<OP>>     \
    operator OPERATOR(const Self & Lhs, const Self & Rhs) {                    \
      return apply<OP>(Lhs, Rhs);                                              \
    }                                                                          \
                                                                               \
    template <typename T = element_type>                                       \
    friend std::enable_if_t<is_op_available_for_type<OP, T>, result_t<OP>>     \
    operator OPERATOR(const Self & Lhs, const element_type & Rhs) {            \
      return OP{}(Lhs, Self{Rhs});                                             \
    }                                                                          \
    template <typename T = element_type>                                       \
    friend std::enable_if_t<is_op_available_for_type<OP, T>, result_t<OP>>     \
    operator OPERATOR(const element_type & Lhs, const Self & Rhs) {            \
      return OP{}(Self{Lhs}, Rhs);                                             \
    }                                                                          \
  };

#define __SYCL_VEC_OPASSIGN_MIXIN(OP, OPERATOR)                                \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OpAssign<OP>>>> {     \
    template <typename T = element_type>                                       \
    friend std::enable_if_t<is_op_available_for_type<OP, T>, Self> &           \
    operator OPERATOR(Self & Lhs, const Self & Rhs) {                          \
      Lhs = OP{}(Lhs, Rhs);                                                    \
      return Lhs;                                                              \
    }                                                                          \
    template <int Num = N, typename T = element_type>                          \
    friend std::enable_if_t<(Num != 1) && (is_op_available_for_type<OP, T>),   \
                            Self &>                                            \
    operator OPERATOR(Self & Lhs, const element_type & Rhs) {                  \
      Lhs = OP{}(Lhs, Self{Rhs});                                              \
      return Lhs;                                                              \
    }                                                                          \
  };

#define __SYCL_VEC_UOP_MIXIN(OP, OPERATOR)                                     \
  template <typename Op>                                                       \
  struct OpMixin<                                                              \
      Op, std::enable_if_t<std::is_same_v<Op, OP> && /* bit_not is handled     \
                                                        separately below */    \
                           !std::is_same_v<Op, std::bit_not<void>>>> {         \
    friend auto operator OPERATOR(const Self &v) { return apply<OP>(v); }      \
  };

#else

#define __SYCL_VEC_BINOP_MIXIN(OP, OPERATOR)                                   \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OP>>> {               \
    friend result_t<OP> operator OPERATOR(const Self & lhs,                    \
                                          const Self & rhs) {                  \
      return VecOperators::apply<OP>(lhs, rhs);                                \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, result_t<OP>>             \
    operator OPERATOR(const Self & lhs, const T & rhs) {                       \
      return VecOperators::apply<OP>(lhs, Self{static_cast<T>(rhs)});          \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, result_t<OP>>             \
    operator OPERATOR(const T & lhs, const Self & rhs) {                       \
      return VecOperators::apply<OP>(Self{static_cast<T>(lhs)}, rhs);          \
    }                                                                          \
  };

#define __SYCL_VEC_OPASSIGN_MIXIN(OP, OPERATOR)                                \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OpAssign<OP>>>> {     \
    friend Self &operator OPERATOR(Self & lhs, const Self & rhs) {             \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, Self &>                   \
    operator OPERATOR(Self & lhs, const T & rhs) {                             \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
  };

#define __SYCL_VEC_UOP_MIXIN(OP, OPERATOR)                                     \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OP>>> {               \
    friend result_t<OP> operator OPERATOR(const Self & v) {                    \
      return apply<OP>(v);                                                     \
    }                                                                          \
  };

#endif

  __SYCL_INSTANTIATE_OPERATORS(__SYCL_VEC_BINOP_MIXIN,
                               __SYCL_VEC_OPASSIGN_MIXIN, __SYCL_VEC_UOP_MIXIN)

#undef __SYCL_VEC_UOP_MIXIN
#undef __SYCL_VEC_OPASSIGN_MIXIN
#undef __SYCL_VEC_BINOP_MIXIN

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
  template <typename Op>
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, std::bit_not<void>>>> {
    template <typename T = typename from_incomplete<Self>::element_type>
    friend std::enable_if_t<is_op_available_for_type<Op, T>, Self>
    operator~(const Self &v) {
      return apply<std::bit_not<void>>(v);
    }
  };
#endif

  template <typename... Op>
  struct __SYCL_EBO CombineImpl : public OpMixin<Op>... {};

  struct Combined
      : CombineImpl<std::plus<void>, std::minus<void>, std::multiplies<void>,
                    std::divides<void>, std::modulus<void>, std::bit_and<void>,
                    std::bit_or<void>, std::bit_xor<void>, std::equal_to<void>,
                    std::not_equal_to<void>, std::less<void>,
                    std::greater<void>, std::less_equal<void>,
                    std::greater_equal<void>, std::logical_and<void>,
                    std::logical_or<void>, ShiftLeft, ShiftRight,
                    std::negate<void>, std::logical_not<void>,
                    std::bit_not<void>, UnaryPlus, OpAssign<std::plus<void>>,
                    OpAssign<std::minus<void>>, OpAssign<std::multiplies<void>>,
                    OpAssign<std::divides<void>>, OpAssign<std::modulus<void>>,
                    OpAssign<std::bit_and<void>>, OpAssign<std::bit_or<void>>,
                    OpAssign<std::bit_xor<void>>, OpAssign<ShiftLeft>,
                    OpAssign<ShiftRight>, IncDec> {};
};

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
template <typename Self> struct SwizzleOperators {
  using element_type = typename from_incomplete<Self>::element_type;
  using vec_ty = typename from_incomplete<Self>::result_vec_ty;
  static constexpr int N = from_incomplete<Self>::size();

  template <typename T>
  static constexpr bool is_compatible_scalar =
      std::is_convertible_v<T, typename from_incomplete<Self>::element_type> &&
      !is_swizzle_v<T>;

  // Can't use partial specialization on constexpr variables because it took too
  // long for gcc to fix https://gcc.gnu.org/bugzilla/show_bug.cgi?id=71954 and
  // we need to support older versions without the fix.
  template <typename OtherSwizzle, typename = void>
  struct is_compatible_swizzle_impl : std::false_type {};

  template <typename OtherSwizzle>
  struct is_compatible_swizzle_impl<
      OtherSwizzle, std::enable_if_t<is_swizzle_v<OtherSwizzle>>>
      : std::bool_constant<
            std::is_same_v<typename from_incomplete<OtherSwizzle>::element_type,
                           typename from_incomplete<Self>::element_type> &&
            from_incomplete<OtherSwizzle>::size() ==
                from_incomplete<Self>::size()> {};

  template <typename OtherSwizzle>
  static constexpr bool is_compatible_swizzle =
      is_compatible_swizzle_impl<OtherSwizzle>::value;

  template <typename OtherSwizzle, typename = void>
  struct is_compatible_swizzle_opposite_const_impl : std::false_type {};

  template <typename OtherSwizzle>
  struct is_compatible_swizzle_opposite_const_impl<
      OtherSwizzle, std::enable_if_t<is_swizzle_v<OtherSwizzle>>>
      : std::bool_constant<is_compatible_swizzle<OtherSwizzle> &&
                           from_incomplete<OtherSwizzle>::is_over_const_vec !=
                               from_incomplete<Self>::is_over_const_vec> {};

  template <typename OtherSwizzle>
  static constexpr bool is_compatible_swizzle_opposite_const =
      is_compatible_swizzle_opposite_const_impl<OtherSwizzle>::value;

  template <typename Op>
  using result_t = std::conditional_t<
      is_logical<Op>, vec<fixed_width_signed<sizeof(element_type)>, N>, vec_ty>;

  // Uglier than possible due to
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282.
  template <typename Op, typename = void> struct OpMixin;

  template <typename Op>
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, IncDec>>>
      : public IncDecImpl<const Self> {};

#define __SYCL_SWIZZLE_BINOP_MIXIN(OP, OPERATOR)                               \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OP>>> {               \
    friend result_t<OP> operator OPERATOR(const Self & lhs,                    \
                                          const vec_ty & rhs) {                \
      return OP{}(vec_ty{lhs}, rhs);                                           \
    }                                                                          \
    friend result_t<OP> operator OPERATOR(const vec_ty & lhs,                  \
                                          const Self & rhs) {                  \
      return OP{}(lhs, vec_ty{rhs});                                           \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, result_t<OP>>             \
    operator OPERATOR(const Self & lhs, const T & rhs) {                       \
      return OP{}(vec_ty{lhs}, vec_ty{rhs});                                   \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, result_t<OP>>             \
    operator OPERATOR(const T & lhs, const Self & rhs) {                       \
      return OP{}(vec_ty{lhs}, vec_ty{rhs});                                   \
    }                                                                          \
    template <typename OtherSwizzle>                                           \
    friend std::enable_if_t<is_compatible_swizzle<OtherSwizzle>, result_t<OP>> \
    operator OPERATOR(const Self & lhs, const OtherSwizzle & rhs) {            \
      return OP{}(vec_ty{lhs}, vec_ty{rhs});                                   \
    }                                                                          \
    template <typename OtherSwizzle>                                           \
    friend std::enable_if_t<                                                   \
        is_compatible_swizzle_opposite_const<OtherSwizzle>, result_t<OP>>      \
    operator OPERATOR(const OtherSwizzle & lhs, const Self & rhs) {            \
      return OP{}(vec_ty{lhs}, vec_ty{rhs});                                   \
    }                                                                          \
  };

#define __SYCL_SWIZZLE_OPASSIGN_MIXIN(OP, OPERATOR)                            \
  template <typename Op>                                                       \
  struct OpMixin<OpAssign<Op>, std::enable_if_t<std::is_same_v<Op, OP>>> {     \
    friend const Self &operator OPERATOR(const Self & lhs,                     \
                                         const vec_ty & rhs) {                 \
      lhs = OP{}(vec_ty{lhs}, rhs);                                            \
      return lhs;                                                              \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<is_compatible_scalar<T>, const Self &>             \
    operator OPERATOR(const Self & lhs, const T & rhs) {                       \
      lhs = OP{}(vec_ty{lhs}, vec_ty{rhs});                                    \
      return lhs;                                                              \
    }                                                                          \
    template <typename OtherSwizzle>                                           \
    friend std::enable_if_t<is_compatible_swizzle<OtherSwizzle>, const Self &> \
    operator OPERATOR(const Self & lhs, const OtherSwizzle & rhs) {            \
      lhs = OP{}(vec_ty{lhs}, vec_ty{rhs});                                    \
      return lhs;                                                              \
    }                                                                          \
  };

#define __SYCL_SWIZZLE_UOP_MIXIN(OP, OPERATOR)                                 \
  template <typename Op>                                                       \
  struct OpMixin<Op, std::enable_if_t<std::is_same_v<Op, OP>>> {               \
    friend result_t<OP> operator OPERATOR(const Self & v) {                    \
      return OP{}(vec_ty{v});                                                  \
    }                                                                          \
  };

  __SYCL_INSTANTIATE_OPERATORS(__SYCL_SWIZZLE_BINOP_MIXIN,
                               __SYCL_SWIZZLE_OPASSIGN_MIXIN,
                               __SYCL_SWIZZLE_UOP_MIXIN)

#undef __SYCL_SWIZZLE_UOP_MIXIN
#undef __SYCL_SWIZZLE_OPASSIGN_MIXIN
#undef __SYCL_SWIZZLE_BINOP_MIXIN

  template <typename... Op>
  struct __SYCL_EBO CombineImpl
      : ApplyIf<is_op_available_for_type<Op, element_type>, OpMixin<Op>>... {};

  template <typename _Self, typename = void>
  struct CombinedImpl
      : CombineImpl<std::plus<void>, std::minus<void>, std::multiplies<void>,
                    std::divides<void>, std::modulus<void>, std::bit_and<void>,
                    std::bit_or<void>, std::bit_xor<void>, std::equal_to<void>,
                    std::not_equal_to<void>, std::less<void>,
                    std::greater<void>, std::less_equal<void>,
                    std::greater_equal<void>, std::logical_and<void>,
                    std::logical_or<void>, ShiftLeft, ShiftRight,
                    std::negate<void>, std::logical_not<void>,
                    std::bit_not<void>, UnaryPlus> {};

  template <typename _Self>
  struct CombinedImpl<_Self,
                      std::enable_if_t<from_incomplete<_Self>::is_assignable>>
      : CombineImpl<std::plus<void>, std::minus<void>, std::multiplies<void>,
                    std::divides<void>, std::modulus<void>, std::bit_and<void>,
                    std::bit_or<void>, std::bit_xor<void>, std::equal_to<void>,
                    std::not_equal_to<void>, std::less<void>,
                    std::greater<void>, std::less_equal<void>,
                    std::greater_equal<void>, std::logical_and<void>,
                    std::logical_or<void>, ShiftLeft, ShiftRight,
                    std::negate<void>, std::logical_not<void>,
                    std::bit_not<void>, UnaryPlus, OpAssign<std::plus<void>>,
                    OpAssign<std::minus<void>>, OpAssign<std::multiplies<void>>,
                    OpAssign<std::divides<void>>, OpAssign<std::modulus<void>>,
                    OpAssign<std::bit_and<void>>, OpAssign<std::bit_or<void>>,
                    OpAssign<std::bit_xor<void>>, OpAssign<ShiftLeft>,
                    OpAssign<ShiftRight>, IncDec> {};

  using Combined = CombinedImpl<Self>;
};
#endif

#if __SYCL_USE_LIBSYCL8_VEC_IMPL
template <typename DataT, int NumElements>
class vec_arith : public VecOperators<vec<DataT, NumElements>>::Combined {};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <int NumElements>
class vec_arith<std::byte, NumElements>
    : public VecOperators<vec<std::byte, NumElements>>::template CombineImpl<
          std::bit_or<void>, std::bit_and<void>, std::bit_xor<void>,
          std::bit_not<void>> {
protected:
  // NumElements can never be zero. Still using the redundant check to avoid
  // incomplete type errors.
  using DataT = typename std::conditional_t<NumElements == 0, int, std::byte>;
  using vec_t = vec<DataT, NumElements>;

  // Special <<, >> operators for std::byte.
  // std::byte is not an arithmetic type and it only supports the following
  // overloads of >> and << operators.
  //
  // 1 template <class IntegerType>
  //   constexpr std::byte operator<<( std::byte b, IntegerType shift )
  //   noexcept;
  friend vec_t operator<<(const vec_t &Lhs, int shift) {
    vec_t Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = Lhs[I] << shift;
    }
    return Ret;
  }
  friend vec_t &operator<<=(vec_t &Lhs, int shift) {
    Lhs = Lhs << shift;
    return Lhs;
  }

  // 2 template <class IntegerType>
  //   constexpr std::byte operator>>( std::byte b, IntegerType shift )
  //   noexcept;
  friend vec_t operator>>(const vec_t &Lhs, int shift) {
    vec_t Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = Lhs[I] >> shift;
    }
    return Ret;
  }
  friend vec_t &operator>>=(vec_t &Lhs, int shift) {
    Lhs = Lhs >> shift;
    return Lhs;
  }
};
#endif // (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
#endif

#undef __SYCL_INSTANTIATE_OPERATORS

} // namespace detail
} // namespace _V1
} // namespace sycl
