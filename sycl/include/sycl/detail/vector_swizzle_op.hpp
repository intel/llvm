//==----------- vector_swizzle_op.hpp - libsycl8 swizzle ops --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/vector_core.hpp>

#include <array>
#include <functional>
#include <utility>

namespace sycl {
inline namespace _V1 {

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

template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
inline simplify_if_swizzle_t<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                       OperationCurrentT, Indexes...>>
materialize_if_swizzle(
    const SwizzleOp<VecT, OperationLeftT, OperationRightT, OperationCurrentT,
                    Indexes...> &X) {
  return simplify_if_swizzle_t<
      SwizzleOp<VecT, OperationLeftT, OperationRightT, OperationCurrentT,
                Indexes...>>{X};
}
///////////////////////// class SwizzleOp /////////////////////////
} // namespace detail
#endif
} // namespace _V1
} // namespace sycl