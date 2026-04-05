//==------------- vector_swizzle.hpp - vec/swizzle support ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/vector_arith.hpp>

#include <sycl/detail/common.hpp>
#include <sycl/detail/fwd/accessor.hpp>
#include <sycl/detail/fwd/half.hpp>

#include <cstddef>
#include <type_traits>

namespace sycl {

// TODO: It should be within _V1 namespace, fix in the next ABI breaking
// windows.
enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {
namespace ext::oneapi {
class bfloat16;
}

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
//   (void)static_cast<bool>(v);
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

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
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

template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
inline simplify_if_swizzle_t<hide_swizzle_from_adl::Swizzle<
    IsConstVec, DataT, VecSize, Indexes...>>
materialize_if_swizzle(const hide_swizzle_from_adl::Swizzle<
                      IsConstVec, DataT, VecSize, Indexes...> &X) {
  return static_cast<simplify_if_swizzle_t<hide_swizzle_from_adl::Swizzle<
      IsConstVec, DataT, VecSize, Indexes...>>>(X);
}
} // namespace hide_swizzle_from_adl
#endif
} // namespace detail
} // namespace _V1
} // namespace sycl