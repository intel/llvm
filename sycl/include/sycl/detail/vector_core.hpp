//==------------- vector_core.hpp --- Implements sycl::vec core ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/vector_traits.hpp>

#include <sycl/bit_cast.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/fwd/accessor.hpp>
#include <sycl/detail/fwd/half.hpp>
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/detail/generic_type_traits.hpp>

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
// To be defined in tests, trick to access vec's private methods
template <typename T1, int T2> class vec_base_test;

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

  template <
      typename T, typename = std::enable_if_t<!std::is_same_v<T, element_type>>,
      typename =
          std::void_t<decltype(static_cast<T>(std::declval<element_type>()))>>
  explicit operator T() const {
    return static_cast<T>((*static_cast<const Self *>(this))[0]);
  }
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
  // Plain array is used by default. Defining
  // __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE to 0 restores the old behavior of
  // using plain array only if its size and alignment match those of std::array.
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
  static constexpr size_t alignment =
      sizeof(DataType) < size_t{64} ? sizeof(DataType) : size_t{64};
  alignas(alignment) DataType m_Data;

  template <size_t... Is>
  constexpr vec_base(const DataT &Val, std::index_sequence<Is...>)
      : m_Data{((void)Is, Val)...} {}

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
      : vec_base(arg, std::make_index_sequence<NumElements>()) {}

  template <typename... argTN,
            typename = std::enable_if_t<
                ((AllowArgTypeInVariadicCtor<argTN> && ...)) &&
                ((num_elements<argTN>() + ...)) == NumElements>>
  constexpr vec_base(const argTN &...args)
      : vec_base{VecArgArrayCreator<DataT, argTN...>::Create(args...),
                 std::make_index_sequence<NumElements>()} {}
};

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
  static constexpr size_t alignment =
      sizeof(DataType) < size_t{64} ? sizeof(DataType) : size_t{64};
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
      public swizzle_ops_base<
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
} // namespace detail

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec
    : public detail::vec_ops_base<vec<DataT, NumElements>>::Combined,
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

  template <int... Indexes>
  using Swizzle =
      detail::hide_swizzle_from_adl::Swizzle<false, DataT, NumElements,
                                             Indexes...>;

  template <int... Indexes>
  using ConstSwizzle =
      detail::hide_swizzle_from_adl::Swizzle<true, DataT, NumElements,
                                             Indexes...>;

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

  template <typename T>
  typename std::enable_if_t<std::is_convertible_v<T, DataT>, vec &>
  operator=(const T &Rhs) {
    *this = vec{static_cast<DataT>(Rhs)};
    return *this;
  }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(Base); }

  // Out-of-class definition is in `sycl/detail/vector_convert.hpp`
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const;

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
    return Swizzle<SwizzleIndexes...>{*this};
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
    return ConstSwizzle<SwizzleIndexes...>{*this};
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
  template <typename T1, int T2> friend class __SYCL_EBO vec;
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
