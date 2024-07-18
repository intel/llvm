//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/access/access.hpp>              // for address_s...
#include <sycl/accessor.hpp>                   // for local_acc...
#include <sycl/aspects.hpp>                    // for aspect
#include <sycl/atomic.hpp>                     // for IsValidAt...
#include <sycl/atomic_ref.hpp>                 // for atomic_ref
#include <sycl/buffer.hpp>                     // for buffer
#include <sycl/builtins.hpp>                   // for min
#include <sycl/detail/export.hpp>              // for __SYCL_EX...
#include <sycl/detail/generic_type_traits.hpp> // for is_sgenfloat
#include <sycl/detail/impl_utils.hpp>          // for createSyc...
#include <sycl/detail/item_base.hpp>           // for id
#include <sycl/detail/reduction_forward.hpp>   // for strategy
#include <sycl/detail/tuple.hpp>               // for make_tuple
#include <sycl/device.hpp>                     // for device
#include <sycl/event.hpp>                      // for event
#include <sycl/exception.hpp>                  // for make_erro...
#include <sycl/exception_list.hpp>             // for queue_impl
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp> // for buffer
#include <sycl/group.hpp>                           // for workGroup...
#include <sycl/group_algorithm.hpp>                 // for reduce_ov...
#include <sycl/handler.hpp>                         // for handler
#include <sycl/id.hpp>                              // for getDeline...
#include <sycl/kernel.hpp>                          // for auto_name
#include <sycl/known_identity.hpp>                  // for IsKnownId...
#include <sycl/marray.hpp>                          // for marray
#include <sycl/memory_enums.hpp>                    // for memory_order
#include <sycl/multi_ptr.hpp>                       // for address_s...
#include <sycl/nd_item.hpp>                         // for nd_item
#include <sycl/nd_range.hpp>                        // for nd_range
#include <sycl/properties/accessor_properties.hpp>  // for no_init
#include <sycl/properties/reduction_properties.hpp> // for initializ...
#include <sycl/property_list.hpp>                   // for property_...
#include <sycl/queue.hpp>                           // for queue
#include <sycl/range.hpp>                           // for range
#include <sycl/sycl_span.hpp>                       // for dynamic_e...
#include <sycl/usm.hpp>                             // for malloc_de...

#include <algorithm>   // for min
#include <array>       // for array
#include <assert.h>    // for assert
#include <cstddef>     // for size_t
#include <memory>      // for shared_ptr
#include <optional>    // for nullopt
#include <stdint.h>    // for uint32_t
#include <string>      // for operator+
#include <tuple>       // for _Swallow_...
#include <type_traits> // for enable_if_t
#include <utility>     // for index_seq...
#include <variant>     // for tuple

namespace sycl {
inline namespace _V1 {
/// Class that is used to represent objects that are passed to user's lambda
/// functions and representing users' reduction variable.
/// The generic version of the class represents those reductions of those
/// types and operations for which the identity value is not known.
/// The View template describes whether the reducer owns its data or not: if
/// View is 'true', then the reducer does not own its data and instead provides
/// a view of data allocated elsewhere (i.e. via a reference or pointer member);
/// if View is 'false', then the reducer owns its data. With the current default
/// reduction algorithm, the top-level reducers that are passed to the user's
/// lambda contain a private copy of the reduction variable, whereas any reducer
/// created by a subscript operator contains a reference to a reduction variable
/// allocated elsewhere. The Subst parameter is an implementation detail and is
/// used to spell out restrictions using 'enable_if'.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View = false, typename Subst = void>
class reducer;

namespace detail {
// This type trait is used to detect if the atomic operation BinaryOperation
// used with operands of the type T is available for using in reduction.
// The order in which the atomic operations are performed may be arbitrary and
// thus may cause different results from run to run even on the same elements
// and on same device. The macro SYCL_REDUCTION_DETERMINISTIC prohibits using
// atomic operations for reduction and helps to produce stable results.
// SYCL_REDUCTION_DETERMINISTIC is a short term solution, which perhaps become
// deprecated eventually and is replaced by a sycl property passed to reduction.
template <typename T, class BinaryOperation>
using IsReduOptForFastAtomicFetch =
#ifdef SYCL_REDUCTION_DETERMINISTIC
    std::bool_constant<false>;
#else
    std::bool_constant<((is_sgenfloat_v<T> && sizeof(T) == 4) ||
                        is_sgeninteger_v<T>) &&
                       IsValidAtomicType<T>::value &&
                       (IsPlus<T, BinaryOperation>::value ||
                        IsMinimum<T, BinaryOperation>::value ||
                        IsMaximum<T, BinaryOperation>::value ||
                        IsBitOR<T, BinaryOperation>::value ||
                        IsBitXOR<T, BinaryOperation>::value ||
                        IsBitAND<T, BinaryOperation>::value)>;
#endif

// This type trait is used to detect if the atomic operation BinaryOperation
// used with operands of the type T is available for using in reduction, in
// addition to the cases covered by "IsReduOptForFastAtomicFetch", if the device
// has the atomic64 aspect. This type trait should only be used if the device
// has the atomic64 aspect.  Note that this type trait is currently a subset of
// IsReduOptForFastReduce. The macro SYCL_REDUCTION_DETERMINISTIC prohibits
// using the reduce_over_group() algorithm to produce stable results across same
// type devices.
template <typename T, class BinaryOperation>
using IsReduOptForAtomic64Op =
#ifdef SYCL_REDUCTION_DETERMINISTIC
    std::bool_constant<false>;
#else
    std::bool_constant<(IsPlus<T, BinaryOperation>::value ||
                        IsMinimum<T, BinaryOperation>::value ||
                        IsMaximum<T, BinaryOperation>::value) &&
                       is_sgenfloat_v<T> && sizeof(T) == 8>;
#endif

// This type trait is used to detect if the group algorithm reduce() used with
// operands of the type T and the operation BinaryOperation is available
// for using in reduction.
// The macro SYCL_REDUCTION_DETERMINISTIC prohibits using the reduce() algorithm
// to produce stable results across same type devices.
template <typename T, class BinaryOperation>
using IsReduOptForFastReduce =
#ifdef SYCL_REDUCTION_DETERMINISTIC
    std::bool_constant<false>;
#else
    std::bool_constant<((is_sgeninteger_v<T> &&
                         (sizeof(T) == 4 || sizeof(T) == 8)) ||
                        is_sgenfloat_v<T>) &&
                       (IsPlus<T, BinaryOperation>::value ||
                        IsMinimum<T, BinaryOperation>::value ||
                        IsMaximum<T, BinaryOperation>::value)>;
#endif

// std::tuple seems to be a) too heavy and b) not copyable to device now
// Thus sycl::detail::tuple is used instead.
// Switching from sycl::device::tuple to std::tuple can be done by re-defining
// the ReduTupleT type and makeReduTupleT() function below.
template <typename... Ts> using ReduTupleT = sycl::detail::tuple<Ts...>;
template <typename... Ts> ReduTupleT<Ts...> makeReduTupleT(Ts... Elements) {
  return sycl::detail::make_tuple(Elements...);
}

__SYCL_EXPORT size_t reduGetMaxWGSize(std::shared_ptr<queue_impl> Queue,
                                      size_t LocalMemBytesPerWorkItem);
__SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                       size_t &NWorkGroups);
__SYCL_EXPORT size_t reduGetPreferredWGSize(std::shared_ptr<queue_impl> &Queue,
                                            size_t LocalMemBytesPerWorkItem);

template <typename T, class BinaryOperation, bool IsOptional>
class ReducerElement;

/// Helper class for accessing reducer-defined types in CRTP
/// May prove to be useful for other things later
template <typename Reducer> struct ReducerTraits;

template <typename T, class BinaryOperation, int Dims, std::size_t Extent,
          typename IdentityContainerT, bool View, typename Subst>
struct ReducerTraits<reducer<T, BinaryOperation, Dims, Extent,
                             IdentityContainerT, View, Subst>> {
  using type = T;
  using op = BinaryOperation;
  static constexpr int dims = Dims;
  static constexpr size_t extent = Extent;
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type = ReducerElement<T, BinaryOperation, !has_identity>;
};

/// Helper class for accessing internal reducer member functions.
template <typename ReducerT> class ReducerAccess {
public:
  ReducerAccess(ReducerT &ReducerRef) : MReducerRef(ReducerRef) {}

  template <typename ReducerRelayT = ReducerT> auto &getElement(size_t E) {
    return MReducerRef.getElement(E);
  }

  template <typename ReducerRelayT = ReducerT> constexpr auto getIdentity() {
    static_assert(ReducerTraits<ReducerRelayT>::has_identity,
                  "Identity unavailable.");
    return MReducerRef.getIdentity();
  }

  // MSVC does not like static overloads of non-static functions, even if they
  // are made mutually exclusive through SFINAE. Instead we use a new static
  // function to be used when a static function is needed.
  template <typename ReducerRelayT = ReducerT>
  static constexpr auto getIdentityStatic() {
    static_assert(
        IsKnownIdentityOp<typename ReducerTraits<ReducerRelayT>::type,
                          typename ReducerTraits<ReducerRelayT>::op>::value,
        "Static identity unavailable.");
    return ReducerT::getIdentity();
  }

private:
  ReducerT &MReducerRef;
};

// Helper function to simplify the use of ReducerAccess. This avoids the need
// for potentially unsupported deduction guides.
template <typename ReducerT> auto getReducerAccess(ReducerT &Reducer) {
  return ReducerAccess<ReducerT>{Reducer};
}

/// Use CRTP to avoid redefining shorthand operators in terms of combine
///
/// Also, for many types with known identity the operation 'atomic_combine()'
/// is implemented here, which allows to use more efficient version of kernels
/// using those operations, which are based on functionality provided by
/// sycl::atomic class.
///
/// For example, it is known that 0 is identity for sycl::plus operations
/// accepting native scalar types to which scalar 0 is convertible.
/// Also, for int32/64 types the atomic_combine() is lowered to
/// sycl::atomic::fetch_add().
template <class Reducer> class combiner {
  using Ty = typename ReducerTraits<Reducer>::type;
  using BinaryOp = typename ReducerTraits<Reducer>::op;
  static constexpr int Dims = ReducerTraits<Reducer>::dims;
  static constexpr size_t Extent = ReducerTraits<Reducer>::extent;

public:
  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsPlus<_T, BinaryOp>::value &&
                       is_geninteger_v<_T>,
                   Reducer &>
  operator++() {
    return static_cast<Reducer *>(this)->combine(static_cast<_T>(1));
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsPlus<_T, BinaryOp>::value &&
                       is_geninteger_v<_T>,
                   Reducer &>
  operator++(int) {
    return static_cast<Reducer *>(this)->combine(static_cast<_T>(1));
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsPlus<_T, BinaryOp>::value, Reducer &>
  operator+=(const _T &Partial) {
    return static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsMultiplies<_T, BinaryOp>::value, Reducer &>
  operator*=(const _T &Partial) {
    return static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsBitOR<_T, BinaryOp>::value, Reducer &>
  operator|=(const _T &Partial) {
    return static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsBitXOR<_T, BinaryOp>::value, Reducer &>
  operator^=(const _T &Partial) {
    return static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = Ty, int _Dims = Dims>
  std::enable_if_t<(_Dims == 0) && IsBitAND<_T, BinaryOp>::value, Reducer &>
  operator&=(const _T &Partial) {
    return static_cast<Reducer *>(this)->combine(Partial);
  }

private:
  template <access::address_space Space>
  static constexpr memory_scope getMemoryScope() {
    return Space == access::address_space::local_space
               ? memory_scope::work_group
               : memory_scope::device;
  }

  template <access::address_space Space, class T, class AtomicFunctor>
  void atomic_combine_impl(T *ReduVarPtr, AtomicFunctor Functor) const {
    auto reducer = static_cast<const Reducer *>(this);
    for (size_t E = 0; E < Extent; ++E) {
      const auto &ReducerElem = getReducerAccess(*reducer).getElement(E);

      // If the reducer element doesn't have a value we can skip the combine.
      if (!ReducerElem)
        continue;

      auto AtomicRef = sycl::atomic_ref<T, memory_order::relaxed,
                                        getMemoryScope<Space>(), Space>(
          address_space_cast<Space, access::decorated::no>(ReduVarPtr)[E]);
      Functor(std::move(AtomicRef), *ReducerElem);
    }
  }

  template <class _T, access::address_space Space, class BinaryOp>
  static constexpr bool BasicCheck =
      std::is_same_v<remove_decoration_t<_T>, Ty> &&
      (Space == access::address_space::global_space ||
       Space == access::address_space::local_space);

public:
  /// Atomic ADD operation: *ReduVarPtr += MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
                   (IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value ||
                    IsReduOptForAtomic64Op<_T, _BinaryOperation>::value) &&
                   IsPlus<_T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_add(Val); });
  }

  /// Atomic BITWISE OR operation: *ReduVarPtr |= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
                   IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value &&
                   IsBitOR<_T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_or(Val); });
  }

  /// Atomic BITWISE XOR operation: *ReduVarPtr ^= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
                   IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value &&
                   IsBitXOR<_T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_xor(Val); });
  }

  /// Atomic BITWISE AND operation: *ReduVarPtr &= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<std::is_same_v<remove_decoration_t<_T>, _T> &&
                   IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value &&
                   IsBitAND<_T, _BinaryOperation>::value &&
                   (Space == access::address_space::global_space ||
                    Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_and(Val); });
  }

  /// Atomic MIN operation: *ReduVarPtr = sycl::minimum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
                   (IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value ||
                    IsReduOptForAtomic64Op<_T, _BinaryOperation>::value) &&
                   IsMinimum<_T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_min(Val); });
  }

  /// Atomic MAX operation: *ReduVarPtr = sycl::maximum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = Ty, class _BinaryOperation = BinaryOp>
  std::enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
                   (IsReduOptForFastAtomicFetch<_T, _BinaryOperation>::value ||
                    IsReduOptForAtomic64Op<_T, _BinaryOperation>::value) &&
                   IsMaximum<_T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto &&Ref, auto Val) { return Ref.fetch_max(Val); });
  }
};

/// Templated class for common functionality of all reduction implementation
/// classes.
template <typename T, class BinaryOperation, bool ExplicitIdentity,
          typename CondT = void>
class ReductionIdentityContainer {
  static_assert(!std::is_same_v<T, T>,
                "Partial specializations don't cover all possible options!");
};

// Specialization for reductions with explicit identity.
template <typename T, class BinaryOperation, bool ExplicitIdentity>
class ReductionIdentityContainer<
    T, BinaryOperation, ExplicitIdentity,
    enable_if_t<IsKnownIdentityOp<T, BinaryOperation>::value>> {
public:
  static constexpr bool has_identity = true;

  ReductionIdentityContainer(const T &) {}
  ReductionIdentityContainer() {}

  /// Returns the statically known identity value.
  static constexpr T getIdentity() {
    return known_identity_impl<BinaryOperation, T>::value;
  }
};

// Specialization for reductions with explicit identity.
template <typename T, class BinaryOperation>
class ReductionIdentityContainer<
    T, BinaryOperation, true,
    enable_if_t<!IsKnownIdentityOp<T, BinaryOperation>::value>> {
public:
  static constexpr bool has_identity = true;

  ReductionIdentityContainer(const T &Identity) : MIdentity(Identity) {}

  /// Returns the identity value given by user.
  T getIdentity() const { return MIdentity; }

private:
  /// Identity of the BinaryOperation.
  /// The result of BinaryOperation(X, MIdentity) is equal to X for any X.
  const T MIdentity;
};

// Specialization for identityless reductions.
template <typename T, class BinaryOperation>
class ReductionIdentityContainer<
    T, BinaryOperation, false,
    std::enable_if_t<!IsKnownIdentityOp<T, BinaryOperation>::value>> {
public:
  static constexpr bool has_identity = false;
};

// Class representing a single reducer element. In cases where the identity is
// unknown this is allowed to not contain a value.
template <typename T, class BinaryOperation, bool IsOptional>
class ReducerElement {
  using value_type = std::conditional_t<IsOptional, std::optional<T>, T>;

  template <bool ExplicitIdentity>
  constexpr value_type GetInitValue(
      const ReductionIdentityContainer<T, BinaryOperation, ExplicitIdentity>
          &IdentityContainer) {
    constexpr bool ContainerHasIdentity =
        ReductionIdentityContainer<T, BinaryOperation,
                                   ExplicitIdentity>::has_identity;
    static_assert(IsOptional || ContainerHasIdentity);
    if constexpr (!ContainerHasIdentity)
      return std::nullopt;
    else
      return IdentityContainer.getIdentity();
  }

public:
  ReducerElement() = default;
  ReducerElement(T Value) : MValue{Value} {}

  template <bool ExplicitIdentity>
  ReducerElement(
      const ReductionIdentityContainer<T, BinaryOperation, ExplicitIdentity>
          &IdentityContainer)
      : MValue(GetInitValue(IdentityContainer)) {}

  ReducerElement &combine(BinaryOperation BinOp, const T &OtherValue) {
    if constexpr (IsOptional)
      MValue = MValue ? BinOp(*MValue, OtherValue) : OtherValue;
    else
      MValue = BinOp(MValue, OtherValue);
    return *this;
  }

  ReducerElement &combine(BinaryOperation BinOp, const ReducerElement &Other) {
    if constexpr (IsOptional) {
      if (Other.MValue)
        return combine(BinOp, *Other.MValue);
      // If the other value doesn't have a value it is a no-op.
      return *this;
    } else {
      return combine(BinOp, Other.MValue);
    }
  }

  constexpr T &operator*() noexcept {
    if constexpr (IsOptional)
      return *MValue;
    else
      return MValue;
  }
  constexpr const T &operator*() const noexcept {
    if constexpr (IsOptional)
      return *MValue;
    else
      return MValue;
  }

  constexpr explicit operator bool() const {
    if constexpr (IsOptional)
      return MValue.has_value();
    return true;
  }

private:
  value_type MValue;
};
} // namespace detail

// We explicitly claim std::optional as device-copyable in sycl/types.hpp.
// However, that information isn't propagated to the struct/class types that use
// it as a member, so we need to provide this partial specialization as well.
// This is needed to support GNU STL where
// std::is_trivially_copyable_v<std::optional> is false (e.g., 7.5.* ).
template <typename T, class BinaryOperation, bool IsOptional>
struct is_device_copyable<
    detail::ReducerElement<T, BinaryOperation, IsOptional>>
    : is_device_copyable<std::conditional_t<IsOptional, std::optional<T>, T>> {
};

namespace detail {
template <typename T, class BinaryOperation, int Dims> class reducer_common {
public:
  using value_type = T;
  using binary_operation = BinaryOperation;
  static constexpr int dimensions = Dims;
};

// Token class to help with the in-place construction of reducers.
template <class BinaryOperation, typename IdentityContainerT>
struct ReducerToken {
  const IdentityContainerT &IdentityContainer;
  const BinaryOperation BOp;
};

} // namespace detail

/// Specialization of the generic class 'reducer'. It is used for reductions
/// of those types and operations for which the identity value is not statically
/// known but is provided.
///
/// It stores a copy of the identity and binary operation associated with the
/// reduction.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
    std::enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                     !detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public detail::combiner<
          reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
                  std::enable_if_t<
                      Dims == 0 && Extent == 1 && View == false &&
                      !detail::IsKnownIdentityOp<T, BinaryOperation>::value>>>,
      public detail::reducer_common<T, BinaryOperation, Dims> {
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type =
      detail::ReducerElement<T, BinaryOperation, !has_identity>;

public:
  reducer(const IdentityContainerT &IdentityContainer, BinaryOperation BOp)
      : MValue(IdentityContainer), MIdentity(IdentityContainer),
        MBinaryOp(BOp) {}
  reducer(
      const detail::ReducerToken<BinaryOperation, IdentityContainerT> &Token)
      : reducer(Token.IdentityContainer, Token.BOp) {}

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer &combine(const T &Partial) {
    MValue.combine(MBinaryOp, Partial);
    return *this;
  }

  template <bool HasIdentityRelay = has_identity>
  std::enable_if_t<HasIdentityRelay && (HasIdentityRelay == has_identity), T>
  identity() const {
    return MIdentity.getIdentity();
  }

private:
  template <typename ReducerT> friend class detail::ReducerAccess;

  element_type &getElement(size_t) { return MValue; }
  const element_type &getElement(size_t) const { return MValue; }

  detail::ReducerElement<T, BinaryOperation, !has_identity> MValue;
  const IdentityContainerT MIdentity;
  BinaryOperation MBinaryOp;
};

/// Specialization of the generic class 'reducer'. It is used for reductions
/// of those types and operations for which the identity value is known.
///
/// It allows to reduce the size of the 'reducer' object by not holding
/// the identity field inside it and allows to add a default constructor.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
    std::enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                     detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public detail::combiner<
          reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
                  std::enable_if_t<
                      Dims == 0 && Extent == 1 && View == false &&
                      detail::IsKnownIdentityOp<T, BinaryOperation>::value>>>,
      public detail::reducer_common<T, BinaryOperation, Dims> {
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type =
      detail::ReducerElement<T, BinaryOperation, !has_identity>;

public:
  reducer() : MValue(getIdentity()) {}
  reducer(const IdentityContainerT & /* Identity */, BinaryOperation)
      : MValue(getIdentity()) {}
  reducer(
      const detail::ReducerToken<BinaryOperation, IdentityContainerT> &Token)
      : reducer(Token.IdentityContainer, Token.BOp) {}

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer &combine(const T &Partial) {
    BinaryOperation BOp;
    MValue.combine(BOp, Partial);
    return *this;
  }

  T identity() const { return getIdentity(); }

private:
  template <typename ReducerT> friend class detail::ReducerAccess;

  static constexpr T getIdentity() {
    return detail::known_identity_impl<BinaryOperation, T>::value;
  }

  element_type &getElement(size_t) { return MValue; }
  const element_type &getElement(size_t) const { return MValue; }
  detail::ReducerElement<T, BinaryOperation, !has_identity> MValue;
};

/// Component of 'reducer' class for array reductions, representing a single
/// element of the span (as returned by the subscript operator).
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View>
class reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
              std::enable_if_t<Dims == 0 && View == true>>
    : public detail::combiner<
          reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
                  std::enable_if_t<Dims == 0 && View == true>>>,
      public detail::reducer_common<T, BinaryOperation, Dims> {
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type =
      detail::ReducerElement<T, BinaryOperation, !has_identity>;

public:
  reducer(element_type &Ref, BinaryOperation BOp)
      : MElement(Ref), MBinaryOp(BOp) {}
  reducer(
      const detail::ReducerToken<BinaryOperation, IdentityContainerT> &Token)
      : reducer(Token.IdentityContainer, Token.BOp) {}

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer &combine(const T &Partial) {
    MElement.combine(MBinaryOp, Partial);
    return *this;
  }

private:
  template <typename ReducerT> friend class detail::ReducerAccess;

  element_type &getElement(size_t) { return MElement; }
  const element_type &getElement(size_t) const { return MElement; }

  element_type &MElement;
  BinaryOperation MBinaryOp;
};

/// Specialization of 'reducer' class for array reductions exposing the
/// subscript operator.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
    std::enable_if_t<Dims == 1 && View == false &&
                     !detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public detail::combiner<
          reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
                  std::enable_if_t<
                      Dims == 1 && View == false &&
                      !detail::IsKnownIdentityOp<T, BinaryOperation>::value>>>,
      public detail::reducer_common<T, BinaryOperation, Dims> {
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type =
      detail::ReducerElement<T, BinaryOperation, !has_identity>;

public:
  reducer(const IdentityContainerT &IdentityContainer, BinaryOperation BOp)
      : MValue(IdentityContainer), MIdentity(IdentityContainer),
        MBinaryOp(BOp) {}
  reducer(
      const detail::ReducerToken<BinaryOperation, IdentityContainerT> &Token)
      : reducer(Token.IdentityContainer, Token.BOp) {}

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer<T, BinaryOperation, Dims - 1, Extent, IdentityContainerT, true>
  operator[](size_t Index) {
    return {MValue[Index], MBinaryOp};
  }

  template <bool HasIdentityRelay = has_identity>
  std::enable_if_t<HasIdentityRelay && (HasIdentityRelay == has_identity), T>
  identity() const {
    return MIdentity.getIdentity();
  }

private:
  template <typename ReducerT> friend class detail::ReducerAccess;

  element_type &getElement(size_t E) { return MValue[E]; }
  const element_type &getElement(size_t E) const { return MValue[E]; }

  marray<element_type, Extent> MValue;
  const IdentityContainerT MIdentity;
  BinaryOperation MBinaryOp;
};

/// Specialization of 'reducer' class for array reductions accepting a span
/// in cases where the identity value is known.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename IdentityContainerT, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
    std::enable_if_t<Dims == 1 && View == false &&
                     detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public detail::combiner<
          reducer<T, BinaryOperation, Dims, Extent, IdentityContainerT, View,
                  std::enable_if_t<
                      Dims == 1 && View == false &&
                      detail::IsKnownIdentityOp<T, BinaryOperation>::value>>>,
      public detail::reducer_common<T, BinaryOperation, Dims> {
  static constexpr bool has_identity = IdentityContainerT::has_identity;
  using element_type =
      detail::ReducerElement<T, BinaryOperation, !has_identity>;

public:
  reducer() : MValue(getIdentity()) {}
  reducer(const IdentityContainerT & /* Identity */, BinaryOperation)
      : MValue(getIdentity()) {}
  reducer(
      const detail::ReducerToken<BinaryOperation, IdentityContainerT> &Token)
      : reducer(Token.IdentityContainer, Token.BOp) {}

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  // SYCL 2020 revision 4 says this should be const, but this is a bug
  // see https://github.com/KhronosGroup/SYCL-Docs/pull/252
  reducer<T, BinaryOperation, Dims - 1, Extent, IdentityContainerT, true>
  operator[](size_t Index) {
    return {MValue[Index], BinaryOperation()};
  }

  T identity() const { return getIdentity(); }

private:
  template <typename ReducerT> friend class detail::ReducerAccess;

  static constexpr T getIdentity() {
    return detail::known_identity_impl<BinaryOperation, T>::value;
  }

  element_type &getElement(size_t E) { return MValue[E]; }
  const element_type &getElement(size_t E) const { return MValue[E]; }

  marray<element_type, Extent> MValue;
};

namespace detail {

// Used for determining dimensions for temporary storage (mainly).
template <class T> struct data_dim_t {
  static constexpr int value = 1;
};

template <class T, int AccessorDims, access::mode Mode,
          access::placeholder IsPH, typename PropList>
struct data_dim_t<
    accessor<T, AccessorDims, Mode, access::target::device, IsPH, PropList>> {
  static constexpr int value = AccessorDims;
};

template <class T> struct get_red_t;
template <class T> struct get_red_t<T *> {
  using type = T;
};

template <class T, int Dims, typename AllocatorT>
struct get_red_t<buffer<T, Dims, AllocatorT>> {
  using type = T;
};

namespace reduction {
// Kernel name wrapper for initializing reduction-related memory through
// reduction_impl_algo::withInitializedMem.
template <typename KernelName> struct InitMemKrn;
} // namespace reduction

/// A helper to pass undefined (sycl::detail::auto_name) names unmodified. We
/// must do that to avoid name collisions.
template <class KernelName>
using __sycl_init_mem_for =
    std::conditional_t<std::is_same_v<KernelName, auto_name>, auto_name,
                       reduction::InitMemKrn<KernelName>>;

__SYCL_EXPORT void
addCounterInit(handler &CGH, std::shared_ptr<sycl::detail::queue_impl> &Queue,
               std::shared_ptr<int> &Counter);

template <typename T, class BinaryOperation, int Dims, size_t Extent,
          bool ExplicitIdentity, typename RedOutVar>
class reduction_impl_algo {
  using self = reduction_impl_algo<T, BinaryOperation, Dims, Extent,
                                   ExplicitIdentity, RedOutVar>;

  // TODO: Do we also need chooseBinOp?
  static constexpr T chooseIdentity(const T &Identity) {
    // For now the implementation ignores the identity value given by user
    // when the implementation knows the identity.
    // The SPEC could prohibit passing identity parameter to operations with
    // known identity, but that could have some bad consequences too.
    // For example, at some moment the implementation may NOT know the identity
    // for COMPLEX-PLUS reduction. User may create a program that would pass
    // COMPLEX value (0,0) as identity for PLUS reduction. At some later moment
    // when the implementation starts handling COMPLEX-PLUS as known operation
    // the existing user's program remains compilable and working correctly.
    // I.e. with this constructor here, adding more reduction operations to the
    // list of known operations does not break the existing programs.
    if constexpr (is_known_identity) {
      (void)Identity;
      return ReducerAccess<reducer_type>::getIdentityStatic();
    } else {
      return Identity;
    }
  }

public:
  static constexpr bool is_known_identity =
      IsKnownIdentityOp<T, BinaryOperation>::value;
  static constexpr bool has_identity = is_known_identity || ExplicitIdentity;

  using identity_container_type =
      ReductionIdentityContainer<T, BinaryOperation, ExplicitIdentity>;
  using reducer_token_type =
      detail::ReducerToken<BinaryOperation, identity_container_type>;
  using reducer_type =
      reducer<T, BinaryOperation, Dims, Extent, identity_container_type>;
  using reducer_element_type =
      typename ReducerTraits<reducer_type>::element_type;
  using result_type = T;
  using binary_operation = BinaryOperation;

  static constexpr size_t dims = Dims;
  static constexpr bool has_float64_atomics =
      IsReduOptForAtomic64Op<T, BinaryOperation>::value;
  static constexpr bool has_fast_atomics =
      IsReduOptForFastAtomicFetch<T, BinaryOperation>::value;
  static constexpr bool has_fast_reduce =
      IsReduOptForFastReduce<T, BinaryOperation>::value;

  static constexpr bool is_usm = std::is_same_v<RedOutVar, T *>;

  static constexpr size_t num_elements = Extent;

  reduction_impl_algo(const T &Identity, BinaryOperation BinaryOp, bool Init,
                      RedOutVar RedOut)
      : MIdentityContainer(chooseIdentity(Identity)), MBinaryOp(BinaryOp),
        InitializeToIdentity(Init), MRedOut(std::move(RedOut)) {}

  template <typename RelayT = T,
            typename RelayBinaryOperation = BinaryOperation>
  reduction_impl_algo(
      BinaryOperation BinaryOp, bool Init, RedOutVar RedOut,
      std::enable_if_t<IsKnownIdentityOp<RelayT, RelayBinaryOperation>::value,
                       int> = 0)
      : MIdentityContainer(ReducerAccess<reducer_type>::getIdentityStatic()),
        MBinaryOp(BinaryOp), InitializeToIdentity(Init),
        MRedOut(std::move(RedOut)) {}

  template <typename RelayT = T,
            typename RelayBinaryOperation = BinaryOperation>
  reduction_impl_algo(
      BinaryOperation BinaryOp, bool Init, RedOutVar RedOut,
      std::enable_if_t<!IsKnownIdentityOp<RelayT, RelayBinaryOperation>::value,
                       int> = 0)
      : MIdentityContainer(), MBinaryOp(BinaryOp), InitializeToIdentity(Init),
        MRedOut(std::move(RedOut)) {}

  auto getReadAccToPreviousPartialReds(handler &CGH) const {
    CGH.addReduction(MOutBufPtr);
    return accessor{*MOutBufPtr, CGH, sycl::read_only};
  }

  template <bool IsOneWG>
  auto getWriteMemForPartialReds(size_t Size, handler &CGH) {
    // If there is only one WG we can avoid creation of temporary buffer with
    // partial sums and write directly into user's reduction variable.
    if constexpr (IsOneWG) {
      return getUserRedVarAccess(CGH);
    } else {
      MOutBufPtr =
          std::make_shared<buffer<reducer_element_type, 1>>(range<1>(Size));
      CGH.addReduction(MOutBufPtr);
      return accessor{*MOutBufPtr, CGH};
    }
  }

  template <class _T = T> auto &getTempBuffer(size_t Size, handler &CGH) {
    auto Buffer = std::make_shared<buffer<_T, 1>>(range<1>(Size));
    CGH.addReduction(Buffer);
    return *Buffer;
  }

  /// Returns an accessor accessing the memory that will hold the reduction
  /// partial sums.
  /// If \p Size is equal to one, then the reduction result is the final and
  /// needs to be written to user's read-write accessor (if there is such).
  /// Otherwise, a new buffer is created and accessor to that buffer is
  /// returned.
  auto getWriteAccForPartialReds(size_t Size, handler &CGH) {
    static_assert(!has_identity || sizeof(reducer_element_type) == sizeof(T),
                  "Unexpected size of reducer element type.");

    // We can only use the output memory directly if it is not USM and we have
    // and identity, i.e. it has a thin element wrapper.
    if constexpr (!is_usm && has_identity) {
      if (Size == 1) {
        auto ReinterpretRedOut =
            MRedOut.template reinterpret<reducer_element_type>();
        return accessor{ReinterpretRedOut, CGH};
      }
    }

    // Create a new output buffer and return an accessor to it.
    //
    // Array reductions are performed element-wise to avoid stack growth.
    MOutBufPtr =
        std::make_shared<buffer<reducer_element_type, 1>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return accessor{*MOutBufPtr, CGH};
  }

  /// Provide \p Func with a properly initialized memory to write the reduction
  /// result to. It can either be original user's reduction variable or a newly
  /// allocated memory initialized with reduction's identity. In the later case,
  /// after the \p Func finishes, update original user's variable accordingly
  /// (i.e., honoring initialize_to_identity property).
  //
  // This currently optimizes for a number of kernel instantiations instead of
  // runtime latency. That might change in future.
  template <typename KernelName, typename FuncTy,
            bool HasIdentity = has_identity>
  std::enable_if_t<HasIdentity> withInitializedMem(handler &CGH, FuncTy Func) {
    // "Template" lambda to ensure that only one type of Func (USM/Buf) is
    // instantiated for the code below.
    auto DoIt = [&](auto &Out) {
      auto RWReduVal = std::make_shared<std::array<T, num_elements>>();
      for (size_t i = 0; i < num_elements; ++i) {
        (*RWReduVal)[i] = decltype(MIdentityContainer)::getIdentity();
      }
      auto Buf = std::make_shared<buffer<T, 1>>(RWReduVal.get()->data(),
                                                range<1>(num_elements));
      Buf->set_final_data();
      accessor Mem{*Buf, CGH};
      Func(Mem);

      reduction::withAuxHandler(CGH, [&](handler &CopyHandler) {
        // MSVC (19.32.31329) has problems compiling the line below when used
        // as a host compiler in c++17 mode (but not in c++latest)
        //   accessor Mem{*Buf, CopyHandler};
        // so use the old-style API.
        auto Mem =
            Buf->template get_access<access::mode::read_write>(CopyHandler);
        // Since this CG is dependent on the one associated with CGH,
        // registering the auxiliary resources here is enough.
        CopyHandler.addReduction(RWReduVal);
        CopyHandler.addReduction(Buf);
        if constexpr (is_usm) {
          // Can't capture whole reduction, copy into distinct variables.
          bool IsUpdateOfUserVar = !initializeToIdentity();
          auto BOp = getBinaryOperation();

          // Don't use constexpr as non-default host compilers (unlike clang)
          // might actually create a capture resulting in binary differences
          // between host/device in lambda captures.
          size_t NElements = num_elements;

          CopyHandler.single_task<__sycl_init_mem_for<KernelName>>([=] {
            for (size_t i = 0; i < NElements; ++i) {
              if (IsUpdateOfUserVar)
                Out[i] = BOp(Out[i], Mem[i]);
              else
                Out[i] = Mem[i];
            }
          });
        } else {
          accessor OutAcc{Out, CGH};
          CopyHandler.copy(Mem, OutAcc);
        }
      });
    };
    if constexpr (is_usm) {
      // Don't dispatch based on initializeToIdentity() as that would lead
      // to two different instantiations of Func.
      DoIt(MRedOut);
    } else {
      if (initializeToIdentity())
        DoIt(MRedOut);
      else
        Func(accessor{MRedOut, CGH});
    }
  }

  // Overload of withInitializedMem for reducer without identity. Initializing
  // to identity is not allowed in this case.
  template <typename KernelName, typename FuncTy,
            bool HasIdentity = has_identity>
  std::enable_if_t<!HasIdentity> withInitializedMem(handler &CGH, FuncTy Func) {
    std::ignore = CGH;
    assert(!initializeToIdentity() &&
           "Initialize to identity not allowed for identity-less reductions.");
    Func(accessor{MRedOut, CGH});
  }

  const identity_container_type &getIdentityContainer() {
    return MIdentityContainer;
  }

  accessor<int, 1, access::mode::read_write, access::target::device,
           access::placeholder::false_t>
  getReadWriteAccessorToInitializedGroupsCounter(handler &CGH) {
    auto CounterMem = std::make_shared<int>(0);
    CGH.addReduction(CounterMem);
    auto CounterBuf = std::make_shared<buffer<int, 1>>(CounterMem.get(), 1);
    CounterBuf->set_final_data();
    CGH.addReduction(CounterBuf);
    return {*CounterBuf, CGH};
  }

  // On discrete (vs. integrated) GPUs it's faster to initialize memory with an
  // extra kernel than copy it from the host.
  auto getGroupsCounterAccDiscrete(handler &CGH) {
    queue q = createSyclObjFromImpl<queue>(CGH.MQueue);
    device Dev = q.get_device();
    auto Deleter = [=](auto *Ptr) { free(Ptr, q); };

    std::shared_ptr<int> Counter(malloc_device<int>(1, q), Deleter);
    CGH.addReduction(Counter);

    addCounterInit(CGH, CGH.MQueue, Counter);

    return Counter.get();
  }

  /// Returns the binary operation associated with the reduction.
  BinaryOperation getBinaryOperation() const { return MBinaryOp; }
  bool initializeToIdentity() const { return InitializeToIdentity; }

  auto getUserRedVarAccess(handler &CGH) {
    std::ignore = CGH;
    if constexpr (is_usm)
      return MRedOut;
    else
      return accessor{MRedOut, CGH};
  }

private:
  // Object holding the identity if available.
  identity_container_type MIdentityContainer;

  // Array reduction is performed element-wise to avoid stack growth, hence
  // 1-dimensional always.
  std::shared_ptr<buffer<reducer_element_type, 1>> MOutBufPtr;

  BinaryOperation MBinaryOp;
  bool InitializeToIdentity;

  /// User's accessor/USM pointer to where the reduction must be written.
  RedOutVar MRedOut;
};

/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          bool ExplicitIdentity, typename RedOutVar>
class reduction_impl
    : private reduction_impl_base,
      public reduction_impl_algo<T, BinaryOperation, Dims, Extent,
                                 ExplicitIdentity, RedOutVar> {
private:
  using algo = reduction_impl_algo<T, BinaryOperation, Dims, Extent,
                                   ExplicitIdentity, RedOutVar>;
  using self = reduction_impl<T, BinaryOperation, Dims, Extent,
                              ExplicitIdentity, RedOutVar>;

public:
  using algo::is_known_identity;
  using algo::is_usm;

  // Only scalar and 1D array reductions are supported by SYCL 2020.
  static_assert(Dims <= 1, "Multi-dimensional reductions are not supported.");

  /// Constructs reduction_impl when no identity is specified. This is only
  /// available when ExplicitIdentity is false.
  template <bool ExplicitIdentityRelay = ExplicitIdentity,
            typename = std::enable_if_t<!ExplicitIdentityRelay>>
  reduction_impl(RedOutVar Var, BinaryOperation BOp,
                 bool InitializeToIdentity = false)
      : algo(BOp, InitializeToIdentity, Var) {
    if constexpr (!is_usm)
      if (Var.size() != 1)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Reduction variable must be a scalar.");
    if constexpr (!is_known_identity)
      if (InitializeToIdentity)
        throw sycl::exception(make_error_code(errc::invalid),
                              "initialize_to_identity property cannot be "
                              "used with identityless reductions.");
  }

  /// Constructs reduction_impl with an explicit identity value. This is only
  /// available when ExplicitIdentity is true.
  template <bool ExplicitIdentityRelay = ExplicitIdentity,
            typename = std::enable_if_t<ExplicitIdentityRelay>>
  reduction_impl(RedOutVar &Var, const T &Identity, BinaryOperation BOp,
                 bool InitializeToIdentity)
      : algo(Identity, BOp, InitializeToIdentity, Var) {
    if constexpr (!is_usm)
      if (Var.size() != 1)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Reduction variable must be a scalar.");
  }
};

template <class BinaryOp, int Dims, size_t Extent, bool ExplicitIdentity,
          typename RedOutVar, typename... RestTy>
auto make_reduction(RedOutVar RedVar, RestTy &&...Rest) {
  return reduction_impl<typename get_red_t<RedOutVar>::type, BinaryOp, Dims,
                        Extent, ExplicitIdentity, RedOutVar>{
      RedVar, std::forward<RestTy>(Rest)...};
}

namespace reduction {
inline void finalizeHandler(handler &CGH) { CGH.finalize(); }
template <class FunctorTy> void withAuxHandler(handler &CGH, FunctorTy Func) {
  event E = CGH.finalize();
  handler AuxHandler(CGH.MQueue, CGH.eventNeeded());
  if (!createSyclObjFromImpl<queue>(CGH.MQueue).is_in_order())
    AuxHandler.depends_on(E);
  AuxHandler.saveCodeLoc(CGH.MCodeLoc);
  Func(AuxHandler);
  CGH.MLastEvent = AuxHandler.finalize();
  return;
}
} // namespace reduction

// This method is used for implementation of parallel_for accepting 1 reduction.
// TODO: remove this method when everything is switched to general algorithm
// implementing arbitrary number of reductions in parallel_for().
/// Copies the final reduction result kept in read-write accessor to user's
/// USM memory.
template <typename KernelName, class Reduction>
void reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu) {
  static_assert(Reduction::is_usm,
                "All implementations using this helper are expected to have "
                "USM reduction, not a buffer-based one.");
  size_t NElements = Reduction::num_elements;
  auto InAcc = Redu.getReadAccToPreviousPartialReds(CGH);
  auto UserVarPtr = Redu.getUserRedVarAccess(CGH);
  bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
  auto BOp = Redu.getBinaryOperation();
  CGH.single_task<KernelName>([=] {
    for (size_t i = 0; i < NElements; ++i) {
      auto Elem = InAcc[i];
      if (IsUpdateOfUserVar)
        UserVarPtr[i] = BOp(UserVarPtr[i], *Elem);
      else
        UserVarPtr[i] = *Elem;
    }
  });
}

namespace reduction {
template <typename KernelName, strategy S, class... Ts> struct MainKrn;
template <typename KernelName, strategy S, class... Ts> struct AuxKrn;
} // namespace reduction

// Tag structs to help creating unique kernels for multi-reduction cases.
struct KernelOneWGTag {};
struct KernelMultipleWGTag {};

/// A helper to pass undefined (sycl::detail::auto_name) names unmodified. We
/// must do that to avoid name collisions.
template <template <typename, reduction::strategy, typename...> class MainOrAux,
          class KernelName, reduction::strategy Strategy, class... Ts>
using __sycl_reduction_kernel =
    std::conditional_t<std::is_same_v<KernelName, auto_name>, auto_name,
                       MainOrAux<KernelName, Strategy, Ts...>>;

// Implementations.

template <reduction::strategy> struct NDRangeReduction;

template <>
struct NDRangeReduction<reduction::strategy::local_atomic_and_atomic_cross_wg> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    static_assert(Reduction::has_identity,
                  "Identityless reductions are not supported by the "
                  "local_atomic_and_atomic_cross_wg strategy.");

    std::ignore = Queue;
    using Name = __sycl_reduction_kernel<
        reduction::MainKrn, KernelName,
        reduction::strategy::local_atomic_and_atomic_cross_wg>;
    Redu.template withInitializedMem<Name>(CGH, [&](auto Out) {
      size_t NElements = Reduction::num_elements;
      local_accessor<typename Reduction::result_type, 1> GroupSum{NElements,
                                                                  CGH};

      CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<1> NDId) {
        // Call user's functions. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer;
        KernelFunc(NDId, Reducer);

        // Work-group cooperates to initialize multiple reduction variables
        auto LID = NDId.get_local_id(0);
        for (size_t E = LID; E < NElements; E += NDId.get_local_range(0)) {
          GroupSum[E] = getReducerAccess(Reducer).getIdentity();
        }
        workGroupBarrier();

        // Each work-item has its own reducer to combine
        Reducer.template atomic_combine<access::address_space::local_space>(
            &GroupSum[0]);

        // Single work-item performs finalization for entire work-group
        // TODO: Opportunity to parallelize across elements
        workGroupBarrier();
        if (LID == 0) {
          for (size_t E = 0; E < NElements; ++E) {
            *getReducerAccess(Reducer).getElement(E) = GroupSum[E];
          }
          Reducer.atomic_combine(&Out[0]);
        }
      });
    });
  }
};

template <>
struct NDRangeReduction<
    reduction::strategy::group_reduce_and_last_wg_detection> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    static_assert(Reduction::has_identity,
                  "Identityless reductions are not supported by the "
                  "group_reduce_and_last_wg_detection strategy.");

    std::ignore = Queue;
    size_t NElements = Reduction::num_elements;
    size_t WGSize = NDRange.get_local_range().size();
    size_t NWorkGroups = NDRange.get_group_range().size();

    auto Out = Redu.getUserRedVarAccess(CGH);

    auto &PartialSumsBuf = Redu.getTempBuffer(NWorkGroups * NElements, CGH);
    accessor PartialSums(PartialSumsBuf, CGH, sycl::read_write, sycl::no_init);

    bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
    auto Rest = [&](auto NWorkGroupsFinished) {
      local_accessor<int, 1> DoReducePartialSumsInLastWG{1, CGH};

      using Name = __sycl_reduction_kernel<
          reduction::MainKrn, KernelName,
          reduction::strategy::group_reduce_and_last_wg_detection,
          decltype(NWorkGroupsFinished)>;

      CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<1> NDId) {
        // Call user's functions. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer;
        KernelFunc(NDId, Reducer);

        typename Reduction::binary_operation BOp;
        auto Group = NDId.get_group();

        // If there are multiple values, reduce each separately
        // reduce_over_group is only defined for each T, not for span<T, ...>
        size_t LID = NDId.get_local_id(0);
        for (size_t E = 0; E < NElements; ++E) {
          auto &RedElem = *getReducerAccess(Reducer).getElement(E);
          RedElem = reduce_over_group(Group, RedElem, BOp);
          if (LID == 0) {
            if (NWorkGroups == 1) {
              // Can avoid using partial sum and write the final result
              // immediately.
              if (IsUpdateOfUserVar)
                RedElem = BOp(RedElem, Out[E]);
              Out[E] = RedElem;
            } else {
              PartialSums[NDId.get_group_linear_id() * NElements + E] =
                  *getReducerAccess(Reducer).getElement(E);
            }
          }
        }

        if (NWorkGroups == 1)
          // We're done.
          return;

        // Signal this work-group has finished after all values are reduced. We
        // had an implicit work-group barrier in reduce_over_group and all the
        // work since has been done in (LID == 0) work-item, so no extra sync is
        // needed.
        if (LID == 0) {
          auto NFinished =
              sycl::atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                               access::address_space::global_space>(
                  NWorkGroupsFinished[0]);
          DoReducePartialSumsInLastWG[0] =
              ++NFinished == static_cast<int>(NWorkGroups);
        }

        workGroupBarrier();
        if (DoReducePartialSumsInLastWG[0]) {
          // Reduce each result separately
          // TODO: Opportunity to parallelize across elements.
          for (size_t E = 0; E < NElements; ++E) {
            auto LocalSum = getReducerAccess(Reducer).getIdentity();
            for (size_t I = LID; I < NWorkGroups; I += WGSize)
              LocalSum = BOp(LocalSum, PartialSums[I * NElements + E]);
            auto Result = reduce_over_group(Group, LocalSum, BOp);

            if (LID == 0) {
              if (IsUpdateOfUserVar)
                Result = BOp(Result, Out[E]);
              Out[E] = Result;
            }
          }
        }
      });
    };

    auto device = getDeviceFromHandler(CGH);
    // Integrated/discrete GPUs have different faster path. For discrete GPUs
    // fast path requires USM device allocations though, so check for that as
    // well.
    if (device.get_info<info::device::host_unified_memory>() ||
        !device.has(aspect::usm_device_allocations))
      Rest(Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH));
    else
      Rest(Redu.getGroupsCounterAccDiscrete(CGH));
  }
};

/// Computes the greatest power-of-two less than or equal to N.
inline size_t GreatestPowerOfTwo(size_t N) {
  if (N == 0)
    return 0;

  size_t Ret = 1;
  while ((N >>= 1) != 0)
    Ret <<= 1;
  return Ret;
}

template <typename FuncTy>
void doTreeReductionHelper(size_t WorkSize, size_t LID, FuncTy Func) {
  workGroupBarrier();

  // Initial pivot is the greatest power-of-two value smaller or equal to the
  // work size.
  size_t Pivot = GreatestPowerOfTwo(WorkSize);

  // If the pivot is not the same as the work size, it needs to do an initial
  // reduction where we only reduce the N last elements into the first N
  // elements, where N is WorkSize - Pivot.
  // 0                            Pivot                   WorkSize  Power of two
  // |                              |                        |      |
  // +-----------------------+------+------------------------+------+
  //                         |
  //                WorkSize - Pivot
  if (Pivot != WorkSize) {
    if (Pivot + LID < WorkSize)
      Func(LID, Pivot + LID);
    workGroupBarrier();
  }

  // Now the amount of work must be power-of-two, so do the tree reduction.
  for (size_t CurPivot = Pivot >> 1; CurPivot > 0; CurPivot >>= 1) {
    if (LID < CurPivot)
      Func(LID, CurPivot + LID);
    workGroupBarrier();
  }
}

// Enum for specifying work size guarantees in tree-reduction.
enum class WorkSizeGuarantees { None, Equal, LessOrEqual };

template <WorkSizeGuarantees WSGuarantee, int Dim, typename LocalRedsTy,
          typename BinOpTy, typename AccessFuncTy>
void doTreeReduction(size_t WorkSize, nd_item<Dim> NDIt, LocalRedsTy &LocalReds,
                     BinOpTy &BOp, AccessFuncTy AccessFunc) {
  size_t LID = NDIt.get_local_linear_id();
  size_t AdjustedWorkSize;
  if constexpr (WSGuarantee == WorkSizeGuarantees::LessOrEqual ||
                WSGuarantee == WorkSizeGuarantees::Equal) {
    // If there is less-or-equal number of items and amount of work, we just
    // load the work into the local memory and start reducing. If we know it is
    // equal we can let the optimizer remove the check.
    if (WSGuarantee == WorkSizeGuarantees::Equal || LID < WorkSize)
      LocalReds[LID] = AccessFunc(LID);
    AdjustedWorkSize = WorkSize;
  } else {
    // Otherwise we have no guarantee and we need to first reduce the amount of
    // work to fit into the local memory.
    size_t WGSize = NDIt.get_local_range().size();
    AdjustedWorkSize = std::min(WorkSize, WGSize);
    if (LID < AdjustedWorkSize) {
      auto LocalSum = AccessFunc(LID);
      for (size_t I = LID + WGSize; I < WorkSize; I += WGSize)
        LocalSum = BOp(LocalSum, AccessFunc(I));

      LocalReds[LID] = LocalSum;
    }
  }
  doTreeReductionHelper(AdjustedWorkSize, LID, [&](size_t I, size_t J) {
    LocalReds[I] = BOp(LocalReds[I], LocalReds[J]);
  });
}

// Tree-reduction over tuples of accessors. This assumes that WorkSize is
// less than or equal to the work-group size.
// TODO: For variadics/tuples we don't provide such a high-level abstraction as
// for the scalar case above. Is there some C++ magic to level them?
template <typename... LocalAccT, typename... BOPsT, size_t... Is>
void doTreeReductionOnTuple(size_t WorkSize, size_t LID,
                            ReduTupleT<LocalAccT...> &LocalAccs,
                            ReduTupleT<BOPsT...> &BOPs,
                            std::index_sequence<Is...>) {
  doTreeReductionHelper(WorkSize, LID, [&](size_t I, size_t J) {
    auto ProcessOne = [=](auto &LocalAcc, auto &BOp) {
      LocalAcc[I] = BOp(LocalAcc[I], LocalAcc[J]);
    };
    (ProcessOne(std::get<Is>(LocalAccs), std::get<Is>(BOPs)), ...);
  });
}

template <> struct NDRangeReduction<reduction::strategy::range_basic> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    using reducer_type = typename Reduction::reducer_type;
    using element_type = typename ReducerTraits<reducer_type>::element_type;

    // If reduction has an identity and is not USM, the reducer element is just
    // a thin wrapper around the result type so the partial sum will use the
    // output memory iff NWorkGroups == 1. Otherwise, we need to make sure the
    // right output buffer is written in case NWorkGroups == 1.
    constexpr bool UsePartialSumForOutput =
        !Reduction::is_usm && Reduction::has_identity;

    std::ignore = Queue;
    size_t NElements = Reduction::num_elements;
    size_t WGSize = NDRange.get_local_range().size();
    size_t NWorkGroups = NDRange.get_group_range().size();

    bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
    auto PartialSums =
        Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);
    auto Out = [&]() {
      if constexpr (UsePartialSumForOutput)
        return (NWorkGroups == 1)
                   ? PartialSums
                   : Redu.getWriteAccForPartialReds(NElements, CGH);
      else
        return Redu.getUserRedVarAccess(CGH);
    }();
    local_accessor<element_type, 1> LocalReds{WGSize, CGH};
    auto NWorkGroupsFinished =
        Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH);
    local_accessor<int, 1> DoReducePartialSumsInLastWG{1, CGH};

    auto IdentityContainer = Redu.getIdentityContainer();
    auto BOp = Redu.getBinaryOperation();

    using Name = __sycl_reduction_kernel<reduction::MainKrn, KernelName,
                                         reduction::strategy::range_basic>;

    CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<1> NDId) {
      // Call user's functions. Reducer.MValue gets initialized there.
      reducer_type Reducer = reducer_type(IdentityContainer, BOp);
      KernelFunc(NDId, Reducer);

      auto ElementCombiner = [&](element_type &LHS, const element_type &RHS) {
        return LHS.combine(BOp, RHS);
      };

      // If there are multiple values, reduce each separately
      // This prevents local memory from scaling with elements
      size_t LID = NDId.get_local_linear_id();
      for (size_t E = 0; E < NElements; ++E) {

        doTreeReduction<WorkSizeGuarantees::Equal>(
            WGSize, NDId, LocalReds, ElementCombiner,
            [&](size_t) { return getReducerAccess(Reducer).getElement(E); });

        if (LID == 0) {
          auto V = LocalReds[0];

          bool IsOneWG = NWorkGroups == 1;
          if (IsOneWG && IsUpdateOfUserVar)
            V.combine(BOp, Out[E]);

          // if NWorkGroups == 1 && UsePartialSumForOutput, then PartialsSum
          // and Out point to same memory.
          if (UsePartialSumForOutput || !IsOneWG)
            PartialSums[NDId.get_group_linear_id() * NElements + E] = V;
          else if (V)
            Out[E] = *V;
        }
      }

      // Signal this work-group has finished after all values are reduced. We
      // had an implicit work-group barrier in doTreeReduction and all the
      // work since has been done in (LID == 0) work-item, so no extra sync is
      // needed.
      if (LID == 0) {
        auto NFinished =
            sycl::atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                             access::address_space::global_space>(
                NWorkGroupsFinished[0]);
        DoReducePartialSumsInLastWG[0] =
            ++NFinished == NWorkGroups && NWorkGroups > 1;
      }

      workGroupBarrier();
      if (DoReducePartialSumsInLastWG[0]) {
        // Reduce each result separately
        // TODO: Opportunity to parallelize across elements
        for (size_t E = 0; E < NElements; ++E) {
          doTreeReduction<WorkSizeGuarantees::None>(
              NWorkGroups, NDId, LocalReds, ElementCombiner,
              [&](size_t I) { return PartialSums[I * NElements + E]; });
          if (LID == 0) {
            auto V = LocalReds[0];
            if (IsUpdateOfUserVar)
              V.combine(BOp, Out[E]);
            Out[E] = *V;
          }
        }
      }
    });
  }
};

template <>
struct NDRangeReduction<reduction::strategy::group_reduce_and_atomic_cross_wg> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    static_assert(Reduction::has_identity,
                  "Identityless reductions are not supported by the "
                  "group_reduce_and_atomic_cross_wg strategy.");

    std::ignore = Queue;
    using Name = __sycl_reduction_kernel<
        reduction::MainKrn, KernelName,
        reduction::strategy::group_reduce_and_atomic_cross_wg>;
    Redu.template withInitializedMem<Name>(CGH, [&](auto Out) {
      size_t NElements = Reduction::num_elements;

      CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<Dims> NDIt) {
        // Call user's function. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer;
        KernelFunc(NDIt, Reducer);

        typename Reduction::binary_operation BOp;
        for (size_t E = 0; E < NElements; ++E) {
          auto &ReducerElem = getReducerAccess(Reducer).getElement(E);
          *ReducerElem = reduce_over_group(NDIt.get_group(), *ReducerElem, BOp);
        }
        if (NDIt.get_local_linear_id() == 0)
          Reducer.atomic_combine(&Out[0]);
      });
    });
  }
};

template <>
struct NDRangeReduction<
    reduction::strategy::local_mem_tree_and_atomic_cross_wg> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    using reducer_type = typename Reduction::reducer_type;
    using element_type = typename ReducerTraits<reducer_type>::element_type;

    std::ignore = Queue;
    using Name = __sycl_reduction_kernel<
        reduction::MainKrn, KernelName,
        reduction::strategy::local_mem_tree_and_atomic_cross_wg>;
    Redu.template withInitializedMem<Name>(CGH, [&](auto Out) {
      size_t NElements = Reduction::num_elements;
      size_t WGSize = NDRange.get_local_range().size();

      // Use local memory to reduce elements in work-groups into zero-th
      // element.
      local_accessor<element_type, 1> LocalReds{WGSize, CGH};

      CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<Dims> NDIt) {
        // Call user's functions. Reducer.MValue gets initialized there.
        reducer_type Reducer;
        KernelFunc(NDIt, Reducer);

        size_t WGSize = NDIt.get_local_range().size();
        size_t LID = NDIt.get_local_linear_id();

        typename Reduction::binary_operation BOp;
        auto ElementCombiner = [&](element_type &LHS, const element_type &RHS) {
          return LHS.combine(BOp, RHS);
        };

        // If there are multiple values, reduce each separately
        // This prevents local memory from scaling with elements
        for (size_t E = 0; E < NElements; ++E) {

          doTreeReduction<WorkSizeGuarantees::Equal>(
              WGSize, NDIt, LocalReds, ElementCombiner,
              [&](size_t) { return getReducerAccess(Reducer).getElement(E); });

          if (LID == 0)
            getReducerAccess(Reducer).getElement(E) = LocalReds[0];

          // Ensure item 0 is finished with LocalReds before next iteration
          if (E != NElements - 1) {
            NDIt.barrier();
          }
        }

        if (LID == 0) {
          Reducer.atomic_combine(&Out[0]);
        }
      });
    });
  }
};

template <>
struct NDRangeReduction<
    reduction::strategy::group_reduce_and_multiple_kernels> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    static_assert(Reduction::has_identity,
                  "Identityless reductions are not supported by the "
                  "group_reduce_and_multiple_kernels strategy.");

    // Before running the kernels, check that device has enough local memory
    // to hold local arrays that may be required for the reduction algorithm.
    // TODO: If the work-group-size is limited by the local memory, then
    // a special version of the main kernel may be created. The one that would
    // not use local accessors, which means it would not do the reduction in
    // the main kernel, but simply generate Range.get_global_range.size() number
    // of partial sums, leaving the reduction work to the additional/aux
    // kernels.
    constexpr bool HFR = Reduction::has_fast_reduce;
    size_t OneElemSize = HFR ? 0 : sizeof(typename Reduction::result_type);
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it may be safer to use queries to the kernel compiled
    // for the device.
    size_t MaxWGSize = reduGetMaxWGSize(Queue, OneElemSize);
    if (NDRange.get_local_range().size() > MaxWGSize)
      throw sycl::exception(make_error_code(errc::nd_range),
                            "The implementation handling parallel_for with"
                            " reduction requires work group size not bigger"
                            " than " +
                                std::to_string(MaxWGSize));

    size_t NElements = Reduction::num_elements;
    size_t NWorkGroups = NDRange.get_group_range().size();
    auto Out = Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);

    bool IsUpdateOfUserVar =
        !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;

    using Name = __sycl_reduction_kernel<
        reduction::MainKrn, KernelName,
        reduction::strategy::group_reduce_and_multiple_kernels>;

    CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<Dims> NDIt) {
      // Call user's functions. Reducer.MValue gets initialized there.
      typename Reduction::reducer_type Reducer;
      KernelFunc(NDIt, Reducer);

      // Compute the partial sum/reduction for the work-group.
      size_t WGID = NDIt.get_group_linear_id();
      typename Reduction::binary_operation BOp;
      for (size_t E = 0; E < NElements; ++E) {
        typename Reduction::result_type PSum;
        PSum = *getReducerAccess(Reducer).getElement(E);
        PSum = reduce_over_group(NDIt.get_group(), PSum, BOp);
        if (NDIt.get_local_linear_id() == 0) {
          if (IsUpdateOfUserVar)
            PSum = BOp(*Out[E], PSum);
          Out[WGID * NElements + E] = PSum;
        }
      }
    });

    reduction::finalizeHandler(CGH);

    // Run the additional kernel as many times as needed to reduce all partial
    // sums into one scalar.

    // TODO: Create a special slow/sequential version of the kernel that would
    // handle the reduction instead of reporting an assert below.
    if (MaxWGSize <= 1)
      throw sycl::exception(make_error_code(errc::nd_range),
                            "The implementation handling parallel_for with "
                            "reduction requires the maximal work group "
                            "size to be greater than 1 to converge. "
                            "The maximal work group size depends on the "
                            "device and the size of the objects passed to "
                            "the reduction.");
    size_t NWorkItems = NDRange.get_group_range().size();
    while (NWorkItems > 1) {
      reduction::withAuxHandler(CGH, [&](handler &AuxHandler) {
        size_t NElements = Reduction::num_elements;
        size_t NWorkGroups;
        size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

        // The last work-group may be not fully loaded with work, or the work
        // group size may be not power of two. Those two cases considered
        // inefficient as they require additional code and checks in the kernel.
        bool HasUniformWG = NWorkGroups * WGSize == NWorkItems;
        if (!Reduction::has_fast_reduce)
          HasUniformWG = HasUniformWG && (WGSize & (WGSize - 1)) == 0;

        // Get read accessor to the buffer that was used as output
        // in the previous kernel.
        auto In = Redu.getReadAccToPreviousPartialReds(AuxHandler);
        auto Out =
            Redu.getWriteAccForPartialReds(NWorkGroups * NElements, AuxHandler);

        using Name = __sycl_reduction_kernel<
            reduction::AuxKrn, KernelName,
            reduction::strategy::group_reduce_and_multiple_kernels>;

        bool IsUpdateOfUserVar = !Reduction::is_usm &&
                                 !Redu.initializeToIdentity() &&
                                 NWorkGroups == 1;
        range<1> GlobalRange = {HasUniformWG ? NWorkItems
                                             : NWorkGroups * WGSize};
        nd_range<1> Range{GlobalRange, range<1>(WGSize)};
        AuxHandler.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
          typename Reduction::binary_operation BOp;
          size_t WGID = NDIt.get_group_linear_id();
          size_t GID = NDIt.get_global_linear_id();

          for (size_t E = 0; E < NElements; ++E) {
            typename Reduction::result_type PSum =
                (HasUniformWG || (GID < NWorkItems))
                    ? *In[GID * NElements + E]
                    : ReducerAccess<typename Reduction::reducer_type>::
                          getIdentityStatic();
            PSum = reduce_over_group(NDIt.get_group(), PSum, BOp);
            if (NDIt.get_local_linear_id() == 0) {
              if (IsUpdateOfUserVar)
                PSum = BOp(*Out[E], PSum);
              Out[WGID * NElements + E] = PSum;
            }
          }
        });
        NWorkItems = NWorkGroups;
      });
    } // end while (NWorkItems > 1)

    if constexpr (Reduction::is_usm) {
      reduction::withAuxHandler(CGH, [&](handler &CopyHandler) {
        reduSaveFinalResultToUserMem<KernelName>(CopyHandler, Redu);
      });
    }
  }
};

template <> struct NDRangeReduction<reduction::strategy::basic> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    using element_type = typename Reduction::reducer_element_type;

    constexpr bool HFR = Reduction::has_fast_reduce;
    size_t OneElemSize = HFR ? 0 : sizeof(element_type);
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it may be safer to use queries to the kernel
    // compiled for the device.
    size_t MaxWGSize = reduGetMaxWGSize(Queue, OneElemSize);
    if (NDRange.get_local_range().size() > MaxWGSize)
      throw sycl::exception(make_error_code(errc::nd_range),
                            "The implementation handling parallel_for with"
                            " reduction requires work group size not bigger"
                            " than " +
                                std::to_string(MaxWGSize));

    size_t NWorkGroups = NDRange.get_group_range().size();

    bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
    std::ignore = IsUpdateOfUserVar;

    // The type of the Out "accessor" differs between scenarios when there is
    // just one WorkGroup and when there are multiple. Use this lambda to write
    // the code just once.
    auto First = [&](auto KernelTag) {
      // We can deduce IsOneWG from the tag type.
      constexpr bool IsOneWG =
          std::is_same_v<std::remove_reference_t<decltype(KernelTag)>,
                         KernelOneWGTag>;

      constexpr size_t NElements = Reduction::num_elements;

      size_t WGSize = NDRange.get_local_range().size();

      auto Out = [&]() {
        if constexpr (IsOneWG)
          return Redu.getUserRedVarAccess(CGH);
        else
          return Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);
      }();

      // Use local memory to reduce elements in work-groups into 0-th element.
      local_accessor<element_type, 1> LocalReds{WGSize, CGH};

      auto BOp = Redu.getBinaryOperation();
      auto IdentityContainer = Redu.getIdentityContainer();

      using Name = __sycl_reduction_kernel<reduction::MainKrn, KernelName,
                                           reduction::strategy::basic,
                                           decltype(KernelTag)>;

      CGH.parallel_for<Name>(NDRange, Properties, [=](nd_item<Dims> NDIt) {
        // Call user's functions. Reducer.MValue gets initialized there.
        typename Reduction::reducer_type Reducer =
            typename Reduction::reducer_type(IdentityContainer, BOp);
        KernelFunc(NDIt, Reducer);

        size_t WGSize = NDIt.get_local_range().size();
        size_t LID = NDIt.get_local_linear_id();

        auto ElementCombiner = [&](element_type &LHS, const element_type &RHS) {
          return LHS.combine(BOp, RHS);
        };

        // If there are multiple values, reduce each separately
        // This prevents local memory from scaling with elements
        for (size_t E = 0; E < NElements; ++E) {

          doTreeReduction<WorkSizeGuarantees::Equal>(
              WGSize, NDIt, LocalReds, ElementCombiner,
              [&](size_t) { return getReducerAccess(Reducer).getElement(E); });

          // Compute the partial sum/reduction for the work-group.
          if (LID == 0) {
            element_type PSum = LocalReds[0];
            if constexpr (IsOneWG) {
              if (IsUpdateOfUserVar)
                PSum.combine(BOp, Out[E]);
              Out[E] = *PSum;
            } else {
              size_t GrID = NDIt.get_group_linear_id();
              Out[GrID * NElements + E] = PSum;
            }
          }

          // Ensure item 0 is finished with LocalReds before next iteration
          if (E != NElements - 1) {
            NDIt.barrier();
          }
        }
      });
    };

    if (NWorkGroups == 1)
      First(KernelOneWGTag{});
    else
      First(KernelMultipleWGTag{});

    reduction::finalizeHandler(CGH);

    // 2. Run the additional kernel as many times as needed to reduce
    // all partial sums into one scalar.

    // TODO: Create a special slow/sequential version of the kernel that would
    // handle the reduction instead of reporting an assert below.
    if (MaxWGSize <= 1)
      throw sycl::exception(make_error_code(errc::nd_range),
                            "The implementation handling parallel_for with "
                            "reduction requires the maximal work group "
                            "size to be greater than 1 to converge. "
                            "The maximal work group size depends on the "
                            "device and the size of the objects passed to "
                            "the reduction.");
    size_t NWorkItems = NDRange.get_group_range().size();
    while (NWorkItems > 1) {
      size_t NWorkGroups;
      size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

      auto Rest = [&](auto KernelTag) {
        reduction::withAuxHandler(CGH, [&](handler &AuxHandler) {
          // We can deduce IsOneWG from the tag type.
          constexpr bool IsOneWG =
              std::is_same_v<std::remove_reference_t<decltype(KernelTag)>,
                             KernelOneWGTag>;

          constexpr size_t NElements = Reduction::num_elements;

          // The last work-group may be not fully loaded with work, or the work
          // group size may be not power of two. Those two cases considered
          // inefficient as they require additional code and checks in the
          // kernel.
          bool HasUniformWG = NWorkGroups * WGSize == NWorkItems;

          // Get read accessor to the buffer that was used as output
          // in the previous kernel.
          auto In = Redu.getReadAccToPreviousPartialReds(AuxHandler);

          auto Out = [&]() {
            if constexpr (IsOneWG)
              return Redu.getUserRedVarAccess(AuxHandler);
            else
              return Redu.getWriteAccForPartialReds(NWorkGroups * NElements,
                                                    AuxHandler);
          }();

          bool UniformPow2WG = HasUniformWG && (WGSize & (WGSize - 1)) == 0;
          // Use local memory to reduce elements in work-groups into 0-th
          // element.
          local_accessor<element_type, 1> LocalReds{WGSize, AuxHandler};

          auto BOp = Redu.getBinaryOperation();
          using Name = __sycl_reduction_kernel<reduction::AuxKrn, KernelName,
                                               reduction::strategy::basic,
                                               decltype(KernelTag)>;

          range<1> GlobalRange = {UniformPow2WG ? NWorkItems
                                                : NWorkGroups * WGSize};
          nd_range<1> Range{GlobalRange, range<1>(WGSize)};
          AuxHandler.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
            size_t WGSize = NDIt.get_local_range().size();
            size_t LID = NDIt.get_local_linear_id();
            size_t GID = NDIt.get_global_linear_id();
            size_t GrID = NDIt.get_group_linear_id();

            // The last work-group may not have enough work for all its items.
            size_t RemainingWorkSize =
                sycl::min(WGSize, NWorkItems - GrID * WGSize);

            auto ElementCombiner = [&](element_type &LHS,
                                       const element_type &RHS) {
              return LHS.combine(BOp, RHS);
            };

            for (size_t E = 0; E < NElements; ++E) {

              doTreeReduction<WorkSizeGuarantees::LessOrEqual>(
                  RemainingWorkSize, NDIt, LocalReds, ElementCombiner,
                  [&](size_t) { return In[GID * NElements + E]; });

              // Compute the partial sum/reduction for the work-group.
              if (LID == 0) {
                element_type PSum = LocalReds[0];
                if constexpr (IsOneWG) {
                  if (IsUpdateOfUserVar)
                    PSum.combine(BOp, Out[E]);
                  Out[E] = *PSum;
                } else {
                  Out[GrID * NElements + E] = PSum;
                }
              }

              // Ensure item 0 is finished with LocalReds before next iteration
              if (E != NElements - 1) {
                NDIt.barrier();
              }
            }
          });
          NWorkItems = NWorkGroups;
        });
      };

      if (NWorkGroups == 1)
        Rest(KernelOneWGTag{});
      else
        Rest(KernelMultipleWGTag{});
    } // end while (NWorkItems > 1)
  }
};

/// For the given 'Reductions' types pack and indices enumerating them this
/// function either creates new temporary accessors for partial sums (if IsOneWG
/// is false) or returns user's accessor/USM-pointer if (IsOneWG is true).
template <bool IsOneWG, typename... Reductions, size_t... Is>
auto createReduOutAccs(size_t NWorkGroups, handler &CGH,
                       std::tuple<Reductions...> &ReduTuple,
                       std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::get<Is>(ReduTuple).template getWriteMemForPartialReds<IsOneWG>(
          NWorkGroups *
              std::tuple_element_t<Is, std::tuple<Reductions...>>::num_elements,
          CGH)...);
}

template <typename OutAccT, typename LocalAccT, typename BOPT,
          typename IdentityContainerT>
auto getLastCombine(OutAccT OutAcc, LocalAccT LocalAcc, BOPT BOP,
                    IdentityContainerT IdentityContainer,
                    bool IsInitializeToIdentity) {
  if constexpr (!IdentityContainerT::has_identity) {
    return BOP(LocalAcc[0], OutAcc[0]);
  } else {
    return BOP(LocalAcc[0], IsInitializeToIdentity
                                ? IdentityContainer.getIdentity()
                                : OutAcc[0]);
  }
}

template <bool IsOneWG, typename... Reductions, typename... OutAccT,
          typename... LocalAccT, typename... BOPsT, typename... Ts,
          size_t... Is>
void writeReduSumsToOutAccs(
    size_t OutAccIndex, ReduTupleT<OutAccT...> OutAccs,
    ReduTupleT<LocalAccT...> LocalAccs, ReduTupleT<BOPsT...> BOPs,
    ReduTupleT<Ts...> IdentityVals,
    std::array<bool, sizeof...(Reductions)> IsInitializeToIdentity,
    std::index_sequence<Is...>) {
  if constexpr (IsOneWG) {
    // Add the initial value of user's variable to the final result.
    // After this we know there will be a value in the 0th element.
    ((std::get<Is>(LocalAccs)[0] = getLastCombine(
          std::get<Is>(OutAccs), std::get<Is>(LocalAccs), std::get<Is>(BOPs),
          std::get<Is>(IdentityVals), IsInitializeToIdentity[Is])),
     ...);
    ((std::get<Is>(OutAccs)[OutAccIndex] = *std::get<Is>(LocalAccs)[0]), ...);
  } else {
    // The partial sums for the work-group are stored in 0-th elements of local
    // accessors. Simply write those sums to output accessors.
    ((std::get<Is>(OutAccs)[OutAccIndex] = std::get<Is>(LocalAccs)[0]), ...);
  }
}

// Concatenate an empty sequence.
constexpr std::index_sequence<> concat_sequences(std::index_sequence<>) {
  return {};
}

// Concatenate a sequence consisting of 1 element.
template <size_t I>
constexpr std::index_sequence<I> concat_sequences(std::index_sequence<I>) {
  return {};
}

// Concatenate two potentially empty sequences.
template <size_t... Is, size_t... Js>
constexpr std::index_sequence<Is..., Js...>
concat_sequences(std::index_sequence<Is...>, std::index_sequence<Js...>) {
  return {};
}

// Concatenate more than 2 sequences.
template <size_t... Is, size_t... Js, class... Rs>
constexpr auto concat_sequences(std::index_sequence<Is...>,
                                std::index_sequence<Js...>, Rs...) {
  return concat_sequences(std::index_sequence<Is..., Js...>{}, Rs{}...);
}

struct IsNonUsmReductionPredicate {
  template <typename T> struct Func {
    static constexpr bool value = !std::remove_pointer_t<T>::is_usm;
  };
};

struct EmptyReductionPredicate {
  template <typename T> struct Func {
    static constexpr bool value = false;
  };
};

template <bool Cond, size_t I> struct FilterElement {
  using type =
      std::conditional_t<Cond, std::index_sequence<I>, std::index_sequence<>>;
};

/// For each index 'I' from the given indices pack 'Is' this function initially
/// creates a number of short index_sequences, where each of such short
/// index sequences is either empty (if the given Functor returns false for the
/// type T[I]) or 1 element 'I' (otherwise). After that this function
/// concatenates those short sequences into one and returns the result sequence.
template <typename... T, typename FunctorT, size_t... Is,
          std::enable_if_t<(sizeof...(Is) > 0), int> Z = 0>
constexpr auto filterSequenceHelper(FunctorT, std::index_sequence<Is...>) {
  return concat_sequences(
      typename FilterElement<FunctorT::template Func<std::tuple_element_t<
                                 Is, std::tuple<T...>>>::value,
                             Is>::type{}...);
}
template <typename... T, typename FunctorT, size_t... Is,
          std::enable_if_t<(sizeof...(Is) == 0), int> Z = 0>
constexpr auto filterSequenceHelper(FunctorT, std::index_sequence<Is...>) {
  return std::index_sequence<>{};
}

/// For each index 'I' from the given indices pack 'Is' this function returns
/// an index sequence consisting of only those 'I's for which the 'FunctorT'
/// applied to 'T[I]' returns true.
template <typename... T, typename FunctorT, size_t... Is>
constexpr auto filterSequence(FunctorT F, std::index_sequence<Is...> Indices) {
  return filterSequenceHelper<T...>(F, Indices);
}

struct IsScalarReduction {
  template <typename Reduction> struct Func {
    static constexpr bool value =
        (Reduction::dims == 0 && Reduction::num_elements == 1);
  };
};

struct IsArrayReduction {
  template <typename Reduction> struct Func {
    static constexpr bool value =
        (Reduction::dims == 1 && Reduction::num_elements >= 1);
  };
};

template <typename ElementType, typename BOPT>
constexpr auto makeAdjustedBOP(BOPT &BOP) {
  return [&](ElementType &LHS, const ElementType &RHS) {
    return LHS.combine(BOP, RHS);
  };
}

template <typename... Reductions, typename... BOPsT, size_t... Is>
constexpr auto makeAdjustedBOPs(ReduTupleT<BOPsT...> &BOPsTuple,
                                std::index_sequence<Is...>) {
  return makeReduTupleT(
      makeAdjustedBOP<typename std::tuple_element_t<
          Is, std::tuple<Reductions...>>::reducer_element_type>(
          std::get<Is>(BOPsTuple))...);
}

template <typename... Reductions, typename... BOPsT>
constexpr auto makeAdjustedBOPs(ReduTupleT<BOPsT...> &BOPsTuple) {
  return makeAdjustedBOPs<Reductions...>(
      BOPsTuple, std::make_index_sequence<sizeof...(Reductions)>{});
}

/// All scalar reductions are processed together; there is one loop of log2(N)
/// steps, and each reduction uses its own storage.
template <bool IsOneWG, typename... Reductions, int Dims, typename... LocalAccT,
          typename... OutAccT, typename... ReducerT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduCGFuncImplScalar(
    nd_item<Dims> NDIt, ReduTupleT<LocalAccT...> LocalAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, std::tuple<ReducerT...> &ReducersTuple,
    ReduTupleT<Ts...> IdentitiesTuple, ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...> ReduIndices) {
  size_t WGSize = NDIt.get_local_range().size();
  size_t LID = NDIt.get_local_linear_id();

  ((std::get<Is>(LocalAccsTuple)[LID] =
        getReducerAccess(std::get<Is>(ReducersTuple)).getElement(0)),
   ...);

  // We apply tree-reduction on reducer elements so we adjust the operations
  // to combine these.
  auto AdjustedBOPsTuple = makeAdjustedBOPs<Reductions...>(BOPsTuple);

  doTreeReductionOnTuple(WGSize, LID, LocalAccsTuple, AdjustedBOPsTuple,
                         ReduIndices);

  // Compute the partial sum/reduction for the work-group.
  if (LID == 0) {
    size_t GrID = NDIt.get_group_linear_id();
    writeReduSumsToOutAccs<IsOneWG, Reductions...>(
        GrID, OutAccsTuple, LocalAccsTuple, AdjustedBOPsTuple, IdentitiesTuple,
        InitToIdentityProps, ReduIndices);
  }
}

/// Each array reduction is processed separately.
template <bool IsOneWG, typename Reduction, int Dims, typename LocalAccT,
          typename OutAccT, typename ReducerT, typename BOPT>
void reduCGFuncImplArrayHelper(nd_item<Dims> NDIt, LocalAccT LocalReds,
                               OutAccT Out, ReducerT &Reducer, BOPT BOp,
                               bool IsInitializeToIdentity) {
  using element_type = typename Reduction::reducer_element_type;

  size_t WGSize = NDIt.get_local_range().size();
  size_t LID = NDIt.get_local_linear_id();

  auto ElementCombiner = [&](element_type &LHS, const element_type &RHS) {
    return LHS.combine(BOp, RHS);
  };

  // If there are multiple values, reduce each separately
  // This prevents local memory from scaling with elements
  auto NElements = Reduction::num_elements;
  for (size_t E = 0; E < NElements; ++E) {
    doTreeReduction<WorkSizeGuarantees::Equal>(
        WGSize, NDIt, LocalReds, ElementCombiner,
        [&](size_t) { return getReducerAccess(Reducer).getElement(E); });

    // Add the initial value of user's variable to the final result.
    if (LID == 0) {
      size_t GrID = NDIt.get_group_linear_id();
      size_t OutIdx = GrID * NElements + E;
      if constexpr (IsOneWG) {
        // If there is only a single work-group, the output will be an actual
        // value rather than a potentially optional value.
        if constexpr (Reduction::has_identity) {
          Out[OutIdx] = *ElementCombiner(LocalReds[0], IsInitializeToIdentity
                                                           ? Reducer.identity()
                                                           : Out[E]);
        } else {
          Out[OutIdx] = *LocalReds[0];
        }
      } else {
        // Otherwise we propagate a potentially optional value.
        Out[OutIdx] = LocalReds[0];
      }
    }

    // Ensure item 0 is finished with LocalReds before next iteration
    if (E != NElements - 1) {
      NDIt.barrier();
    }
  }
}

template <bool IsOneWG, typename... Reductions, int Dims, typename... LocalAccT,
          typename... OutAccT, typename... ReducerT, typename... BOPsT,
          size_t... Is>
void reduCGFuncImplArray(
    nd_item<Dims> NDIt, ReduTupleT<LocalAccT...> LocalAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, std::tuple<ReducerT...> &ReducersTuple,
    ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...>) {
  using ReductionPack = std::tuple<Reductions...>;
  (reduCGFuncImplArrayHelper<IsOneWG, std::tuple_element_t<Is, ReductionPack>>(
       NDIt, std::get<Is>(LocalAccsTuple), std::get<Is>(OutAccsTuple),
       std::get<Is>(ReducersTuple), std::get<Is>(BOPsTuple),
       InitToIdentityProps[Is]),
   ...);
}

namespace reduction::main_krn {
template <class KernelName, class Accessor> struct NDRangeMulti;
} // namespace reduction::main_krn
template <typename KernelName, typename KernelType, int Dims,
          typename PropertiesT, typename... Reductions, size_t... Is>
void reduCGFuncMulti(handler &CGH, KernelType KernelFunc,
                     const nd_range<Dims> &Range, PropertiesT Properties,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...> ReduIndices) {
  size_t WGSize = Range.get_local_range().size();

  // Split reduction sequence into two:
  // 1) Scalar reductions
  // 2) Array reductions
  // This allows us to reuse the existing implementation for scalar reductions
  // and introduce a new implementation for array reductions. Longer term it
  // may make sense to generalize the code such that each phase below applies
  // to all available reduction implementations -- today all reduction classes
  // use the same privatization-based approach, so this is unnecessary.
  IsScalarReduction ScalarPredicate;
  auto ScalarIs = filterSequence<Reductions...>(ScalarPredicate, ReduIndices);

  IsArrayReduction ArrayPredicate;
  auto ArrayIs = filterSequence<Reductions...>(ArrayPredicate, ReduIndices);

  auto LocalAccsTuple = makeReduTupleT(
      local_accessor<typename Reductions::reducer_element_type, 1>{WGSize,
                                                                   CGH}...);

  // The type of the Out "accessor" differs between scenarios when there is just
  // one WorkGroup and when there are multiple. Use this lambda to write the
  // code just once.
  auto Rest = [&](auto KernelTag, auto OutAccsTuple) {
    auto IdentitiesTuple =
        makeReduTupleT(std::get<Is>(ReduTuple).getIdentityContainer()...);
    auto BOPsTuple =
        makeReduTupleT(std::get<Is>(ReduTuple).getBinaryOperation()...);
    std::array InitToIdentityProps{
        std::get<Is>(ReduTuple).initializeToIdentity()...};

    using Name = __sycl_reduction_kernel<reduction::MainKrn, KernelName,
                                         reduction::strategy::multi,
                                         decltype(KernelTag)>;

    CGH.parallel_for<Name>(Range, Properties, [=](nd_item<Dims> NDIt) {
      // We can deduce IsOneWG from the tag type.
      constexpr bool IsOneWG =
          std::is_same_v<std::remove_reference_t<decltype(KernelTag)>,
                         KernelOneWGTag>;

      // Pass all reductions to user's lambda in the same order as supplied
      // Each reducer initializes its own storage
      auto ReduIndices = std::index_sequence_for<Reductions...>();
      auto ReducerTokensTuple =
          std::tuple{typename Reductions::reducer_token_type{
              std::get<Is>(IdentitiesTuple), std::get<Is>(BOPsTuple)}...};
      auto ReducersTuple = std::tuple<typename Reductions::reducer_type...>{
          std::get<Is>(ReducerTokensTuple)...};
      std::apply([&](auto &...Reducers) { KernelFunc(NDIt, Reducers...); },
                 ReducersTuple);

      // Combine and write-back the results of any scalar reductions
      // reduCGFuncImplScalar<Reductions...>(NDIt, LocalAccsTuple, OutAccsTuple,
      // ReducersTuple, IdentitiesTuple, BOPsTuple, InitToIdentityProps,
      // ReduIndices);
      reduCGFuncImplScalar<IsOneWG, Reductions...>(
          NDIt, LocalAccsTuple, OutAccsTuple, ReducersTuple, IdentitiesTuple,
          BOPsTuple, InitToIdentityProps, ScalarIs);

      // Combine and write-back the results of any array reductions
      // These are handled separately to minimize temporary storage and account
      // for the fact that each array reduction may have a different number of
      // elements to reduce (i.e. a different extent).
      reduCGFuncImplArray<IsOneWG, Reductions...>(
          NDIt, LocalAccsTuple, OutAccsTuple, ReducersTuple, BOPsTuple,
          InitToIdentityProps, ArrayIs);
    });
  };

  size_t NWorkGroups = Range.get_group_range().size();
  if (NWorkGroups == 1)
    Rest(KernelOneWGTag{},
         createReduOutAccs<true>(NWorkGroups, CGH, ReduTuple, ReduIndices));
  else
    Rest(KernelMultipleWGTag{},
         createReduOutAccs<false>(NWorkGroups, CGH, ReduTuple, ReduIndices));
}

// TODO: Is this still needed?
template <typename... Reductions, size_t... Is>
void associateReduAccsWithHandler(handler &CGH,
                                  std::tuple<Reductions...> &ReduTuple,
                                  std::index_sequence<Is...>) {
  auto ProcessOne = [&CGH](auto Redu) {
    if constexpr (!decltype(Redu)::is_usm)
      Redu.getUserRedVarAccess(CGH);
  };
  (ProcessOne(std::get<Is>(ReduTuple)), ...);
}

/// All scalar reductions are processed together; there is one loop of log2(N)
/// steps, and each reduction uses its own storage.
template <bool IsOneWG, typename... Reductions, int Dims, typename... LocalAccT,
          typename... InAccT, typename... OutAccT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduAuxCGFuncImplScalar(
    nd_item<Dims> NDIt, size_t LID, size_t GID, size_t RemainingWorkSize,
    ReduTupleT<LocalAccT...> LocalAccsTuple, ReduTupleT<InAccT...> InAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, ReduTupleT<Ts...> IdentitiesTuple,
    ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...> ReduIndices) {
  // The end work-group may have less work than the rest, so we only need to
  // read the value of the elements that still have work left.
  if (LID < RemainingWorkSize)
    ((std::get<Is>(LocalAccsTuple)[LID] = std::get<Is>(InAccsTuple)[GID]), ...);

  // We apply tree-reduction on reducer elements so we adjust the operations
  // to combine these.
  auto AdjustedBOPsTuple = makeAdjustedBOPs<Reductions...>(BOPsTuple);

  doTreeReductionOnTuple(RemainingWorkSize, LID, LocalAccsTuple,
                         AdjustedBOPsTuple, ReduIndices);

  // Compute the partial sum/reduction for the work-group.
  if (LID == 0) {
    size_t GrID = NDIt.get_group_linear_id();
    writeReduSumsToOutAccs<IsOneWG, Reductions...>(
        GrID, OutAccsTuple, LocalAccsTuple, AdjustedBOPsTuple, IdentitiesTuple,
        InitToIdentityProps, ReduIndices);
  }
}

template <bool IsOneWG, typename Reduction, int Dims, typename LocalAccT,
          typename InAccT, typename OutAccT, typename T, typename BOPT>
void reduAuxCGFuncImplArrayHelper(nd_item<Dims> NDIt, size_t LID, size_t GID,
                                  size_t RemainingWorkSize, LocalAccT LocalReds,
                                  InAccT In, OutAccT Out, T IdentityContainer,
                                  BOPT BOp, bool IsInitializeToIdentity) {
  using element_type = typename Reduction::reducer_element_type;
  auto ElementCombiner = [&](element_type &LHS, const element_type &RHS) {
    return LHS.combine(BOp, RHS);
  };

  // If there are multiple values, reduce each separately
  // This prevents local memory from scaling with elements
  auto NElements = Reduction::num_elements;
  for (size_t E = 0; E < NElements; ++E) {
    doTreeReduction<WorkSizeGuarantees::LessOrEqual>(
        RemainingWorkSize, NDIt, LocalReds, ElementCombiner,
        [&](size_t) { return In[GID * NElements + E]; });

    // Add the initial value of user's variable to the final result.
    if (LID == 0) {
      size_t GrID = NDIt.get_group_linear_id();
      size_t OutIdx = GrID * NElements + E;
      if constexpr (IsOneWG) {
        // If there is only a single work-group, the output will be an actual
        // value rather than a potentially optional value.
        if constexpr (Reduction::has_identity) {
          Out[OutIdx] = *ElementCombiner(LocalReds[0],
                                         IsInitializeToIdentity
                                             ? IdentityContainer.getIdentity()
                                             : Out[E]);
        } else {
          Out[OutIdx] = *LocalReds[0];
        }
      } else {
        // Otherwise we propagate a potentially optional value.
        Out[OutIdx] = LocalReds[0];
      }
    }

    // Ensure item 0 is finished with LocalReds before next iteration
    if (E != NElements - 1) {
      NDIt.barrier();
    }
  }
}

template <bool IsOneWG, typename... Reductions, int Dims, typename... LocalAccT,
          typename... InAccT, typename... OutAccT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduAuxCGFuncImplArray(
    nd_item<Dims> NDIt, size_t LID, size_t GID, size_t RemainingWorkSize,
    ReduTupleT<LocalAccT...> LocalAccsTuple, ReduTupleT<InAccT...> InAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, ReduTupleT<Ts...> IdentitiesTuple,
    ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...>) {
  using ReductionPack = std::tuple<Reductions...>;
  (reduAuxCGFuncImplArrayHelper<IsOneWG,
                                std::tuple_element_t<Is, ReductionPack>>(
       NDIt, LID, GID, RemainingWorkSize, std::get<Is>(LocalAccsTuple),
       std::get<Is>(InAccsTuple), std::get<Is>(OutAccsTuple),
       std::get<Is>(IdentitiesTuple), std::get<Is>(BOPsTuple),
       InitToIdentityProps[Is]),
   ...);
}

namespace reduction::aux_krn {
template <class KernelName, class Predicate> struct Multi;
} // namespace reduction::aux_krn
template <typename KernelName, typename KernelType, typename... Reductions,
          size_t... Is>
size_t reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...> ReduIndices) {
  size_t NWorkGroups;
  size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;
  bool HasUniformWG = Pow2WG && (NWorkGroups * WGSize == NWorkItems);

  // Like reduCGFuncImpl, we also have to split out scalar and array reductions
  IsScalarReduction ScalarPredicate;
  auto ScalarIs = filterSequence<Reductions...>(ScalarPredicate, ReduIndices);

  IsArrayReduction ArrayPredicate;
  auto ArrayIs = filterSequence<Reductions...>(ArrayPredicate, ReduIndices);

  auto LocalAccsTuple = makeReduTupleT(
      local_accessor<typename Reductions::reducer_element_type, 1>{WGSize,
                                                                   CGH}...);
  auto InAccsTuple = makeReduTupleT(
      std::get<Is>(ReduTuple).getReadAccToPreviousPartialReds(CGH)...);

  auto IdentitiesTuple =
      makeReduTupleT(std::get<Is>(ReduTuple).getIdentityContainer()...);
  auto BOPsTuple =
      makeReduTupleT(std::get<Is>(ReduTuple).getBinaryOperation()...);
  std::array InitToIdentityProps{
      std::get<Is>(ReduTuple).initializeToIdentity()...};

  // Predicate/OutAccsTuple below have different type depending on us having
  // just a single WG or multiple WGs. Use this lambda to avoid code
  // duplication.
  auto Rest = [&](auto Predicate, auto OutAccsTuple) {
    auto AccReduIndices = filterSequence<Reductions...>(Predicate, ReduIndices);
    associateReduAccsWithHandler(CGH, ReduTuple, AccReduIndices);
    using Name = __sycl_reduction_kernel<reduction::AuxKrn, KernelName,
                                         reduction::strategy::multi,
                                         decltype(Predicate)>;
    // TODO: Opportunity to parallelize across number of elements
    range<1> GlobalRange = {HasUniformWG ? NWorkItems : NWorkGroups * WGSize};
    nd_range<1> Range{GlobalRange, range<1>(WGSize)};
    CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
      // We can deduce IsOneWG from the predicate type.
      constexpr bool IsOneWG =
          std::is_same_v<std::remove_reference_t<decltype(Predicate)>,
                         IsNonUsmReductionPredicate>;

      size_t WGSize = NDIt.get_local_range().size();
      size_t RemainingWorkSize =
          sycl::min(WGSize, NWorkItems - WGSize * NDIt.get_group_linear_id());
      size_t LID = NDIt.get_local_linear_id();
      size_t GID = NDIt.get_global_linear_id();

      // Handle scalar and array reductions
      reduAuxCGFuncImplScalar<IsOneWG, Reductions...>(
          NDIt, LID, GID, RemainingWorkSize, LocalAccsTuple, InAccsTuple,
          OutAccsTuple, IdentitiesTuple, BOPsTuple, InitToIdentityProps,
          ScalarIs);
      reduAuxCGFuncImplArray<IsOneWG, Reductions...>(
          NDIt, LID, GID, RemainingWorkSize, LocalAccsTuple, InAccsTuple,
          OutAccsTuple, IdentitiesTuple, BOPsTuple, InitToIdentityProps,
          ArrayIs);
    });
  };
  if (NWorkGroups == 1)
    Rest(IsNonUsmReductionPredicate{},
         createReduOutAccs<true>(NWorkGroups, CGH, ReduTuple, ReduIndices));
  else
    Rest(EmptyReductionPredicate{},
         createReduOutAccs<false>(NWorkGroups, CGH, ReduTuple, ReduIndices));

  return NWorkGroups;
}

template <typename Reduction> size_t reduGetMemPerWorkItemHelper(Reduction &) {
  return sizeof(typename Reduction::result_type);
}

template <typename Reduction, typename... RestT>
size_t reduGetMemPerWorkItemHelper(Reduction &, RestT... Rest) {
  return sizeof(typename Reduction::result_type) +
         reduGetMemPerWorkItemHelper(Rest...);
}

template <typename... ReductionT, size_t... Is>
size_t reduGetMemPerWorkItem(std::tuple<ReductionT...> &ReduTuple,
                             std::index_sequence<Is...>) {
  return reduGetMemPerWorkItemHelper(std::get<Is>(ReduTuple)...);
}

/// Utility function: for the given tuple \param Tuple the function returns
/// a new tuple consisting of only elements indexed by the index sequence.
template <typename TupleT, std::size_t... Is>
std::tuple<std::tuple_element_t<Is, TupleT>...>
tuple_select_elements(TupleT Tuple, std::index_sequence<Is...>) {
  return {std::get<Is>(std::move(Tuple))...};
}

template <> struct NDRangeReduction<reduction::strategy::multi> {
  template <typename KernelName, int Dims, typename PropertiesT,
            typename... RestT>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  RestT... Rest) {
    std::tuple<RestT...> ArgsTuple(Rest...);
    constexpr size_t NumArgs = sizeof...(RestT);
    auto KernelFunc = std::get<NumArgs - 1>(ArgsTuple);
    auto ReduIndices = std::make_index_sequence<NumArgs - 1>();
    auto ReduTuple = detail::tuple_select_elements(ArgsTuple, ReduIndices);

    size_t LocalMemPerWorkItem = reduGetMemPerWorkItem(ReduTuple, ReduIndices);
    // TODO: currently the maximal work group size is determined for the given
    // queue/device, while it is safer to use queries to the kernel compiled
    // for the device.
    size_t MaxWGSize = reduGetMaxWGSize(Queue, LocalMemPerWorkItem);
    if (NDRange.get_local_range().size() > MaxWGSize)
      throw sycl::exception(make_error_code(errc::nd_range),
                            "The implementation handling parallel_for with"
                            " reduction requires work group size not bigger"
                            " than " +
                                std::to_string(MaxWGSize));

    reduCGFuncMulti<KernelName>(CGH, KernelFunc, NDRange, Properties, ReduTuple,
                                ReduIndices);
    reduction::finalizeHandler(CGH);

    size_t NWorkItems = NDRange.get_group_range().size();
    while (NWorkItems > 1) {
      reduction::withAuxHandler(CGH, [&](handler &AuxHandler) {
        NWorkItems = reduAuxCGFunc<KernelName, decltype(KernelFunc)>(
            AuxHandler, NWorkItems, MaxWGSize, ReduTuple, ReduIndices);
      });
    } // end while (NWorkItems > 1)
  }
};

// Auto-dispatch. Must be the last one.
template <> struct NDRangeReduction<reduction::strategy::auto_select> {
  // Some readability aliases, to increase signal/noise ratio below.
  template <reduction::strategy Strategy>
  using Impl = NDRangeReduction<Strategy>;
  using Strat = reduction::strategy;

  template <typename KernelName, int Dims, typename PropertiesT,
            typename KernelType, typename Reduction>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  Reduction &Redu, KernelType &KernelFunc) {
    auto Delegate = [&](auto Impl) {
      Impl.template run<KernelName>(CGH, Queue, NDRange, Properties, Redu,
                                    KernelFunc);
    };

    if constexpr (Reduction::has_float64_atomics) {
      if (getDeviceFromHandler(CGH).has(aspect::atomic64))
        return Delegate(Impl<Strat::group_reduce_and_atomic_cross_wg>{});

      if constexpr (Reduction::has_fast_reduce)
        return Delegate(Impl<Strat::group_reduce_and_multiple_kernels>{});
      else
        return Delegate(Impl<Strat::basic>{});
    } else if constexpr (Reduction::has_fast_atomics) {
      if constexpr (sizeof(typename Reduction::result_type) == 8) {
        // Both group_reduce_and_atomic_cross_wg and
        // local_mem_tree_and_atomic_cross_wg implicitly require
        // aspect::atomic64 if the result type of the reduction is 64-bit. If
        // the device does not support this, we need to fall back to more
        // reliable strategies.
        if (!getDeviceFromHandler(CGH).has(aspect::atomic64)) {
          if constexpr (Reduction::has_fast_reduce)
            return Delegate(Impl<Strat::group_reduce_and_multiple_kernels>{});
          else
            return Delegate(Impl<Strat::basic>{});
        }
      }

      if constexpr (Reduction::has_fast_reduce) {
        return Delegate(Impl<Strat::group_reduce_and_atomic_cross_wg>{});
      } else {
        return Delegate(Impl<Strat::local_mem_tree_and_atomic_cross_wg>{});
      }
    } else {
      if constexpr (Reduction::has_fast_reduce)
        return Delegate(Impl<Strat::group_reduce_and_multiple_kernels>{});
      else
        return Delegate(Impl<Strat::basic>{});
    }

    assert(false && "Must be unreachable!");
  }
  template <typename KernelName, int Dims, typename PropertiesT,
            typename... RestT>
  static void run(handler &CGH, std::shared_ptr<detail::queue_impl> &Queue,
                  nd_range<Dims> NDRange, PropertiesT &Properties,
                  RestT... Rest) {
    return Impl<Strat::multi>::run<KernelName>(CGH, Queue, NDRange, Properties,
                                               Rest...);
  }
};

template <typename KernelName, reduction::strategy Strategy, int Dims,
          typename PropertiesT, typename... RestT>
void reduction_parallel_for(handler &CGH, nd_range<Dims> NDRange,
                            PropertiesT Properties, RestT... Rest) {
  NDRangeReduction<Strategy>::template run<KernelName>(CGH, CGH.MQueue, NDRange,
                                                       Properties, Rest...);
}

__SYCL_EXPORT uint32_t
reduGetMaxNumConcurrentWorkGroups(std::shared_ptr<queue_impl> Queue);

template <typename KernelName, reduction::strategy Strategy, int Dims,
          typename PropertiesT, typename... RestT>
void reduction_parallel_for(handler &CGH, range<Dims> Range,
                            PropertiesT Properties, RestT... Rest) {
  std::tuple<RestT...> ArgsTuple(Rest...);
  constexpr size_t NumArgs = sizeof...(RestT);
  static_assert(NumArgs > 1, "No reduction!");
  auto KernelFunc = std::get<NumArgs - 1>(ArgsTuple);
  auto ReduIndices = std::make_index_sequence<NumArgs - 1>();
  auto ReduTuple = detail::tuple_select_elements(ArgsTuple, ReduIndices);

  // Before running the kernels, check that device has enough local memory
  // to hold local arrays required for the tree-reduction algorithm.
  size_t OneElemSize = [&]() {
    // Can't use outlined NumArgs due to a bug in gcc 8.4.
    if constexpr (sizeof...(RestT) == 2) {
      using Reduction = std::tuple_element_t<0, decltype(ReduTuple)>;
      constexpr bool IsTreeReduction =
          !Reduction::has_fast_reduce && !Reduction::has_fast_atomics;
      return IsTreeReduction ? sizeof(typename Reduction::result_type) : 0;
    } else {
      return reduGetMemPerWorkItem(ReduTuple, ReduIndices);
    }
  }();

  uint32_t NumConcurrentWorkGroups =
#ifdef __SYCL_REDUCTION_NUM_CONCURRENT_WORKGROUPS
      __SYCL_REDUCTION_NUM_CONCURRENT_WORKGROUPS;
#else
      reduGetMaxNumConcurrentWorkGroups(CGH.MQueue);
#endif

  // TODO: currently the preferred work group size is determined for the given
  // queue/device, while it is safer to use queries to the kernel pre-compiled
  // for the device.
  size_t PrefWGSize = reduGetPreferredWGSize(CGH.MQueue, OneElemSize);

  size_t NWorkItems = Range.size();
  size_t WGSize = std::min(NWorkItems, PrefWGSize);
  size_t NWorkGroups = NWorkItems / WGSize;
  if (NWorkItems % WGSize)
    NWorkGroups++;
  size_t MaxNWorkGroups = NumConcurrentWorkGroups;
  NWorkGroups = std::min(NWorkGroups, MaxNWorkGroups);
  size_t NDRItems = NWorkGroups * WGSize;
  nd_range<1> NDRange{range<1>{NDRItems}, range<1>{WGSize}};

  size_t PerGroup = Range.size() / NWorkGroups;
  // Iterate through the index space by assigning contiguous chunks to each
  // work-group, then iterating through each chunk using a stride equal to the
  // work-group's local range, which gives much better performance than using
  // stride equal to 1. For each of the index the given the original KernelFunc
  // is called and the reduction value hold in \p Reducer is accumulated in
  // those calls.
  auto UpdatedKernelFunc = [=](auto NDId, auto &...Reducers) {
    // Divide into contiguous chunks and assign each chunk to a Group
    // Rely on precomputed division to avoid repeating expensive operations
    // TODO: Some devices may prefer alternative remainder handling
    auto Group = NDId.get_group();
    size_t GroupId = Group.get_group_linear_id();
    size_t NumGroups = Group.get_group_linear_range();
    bool LastGroup = (GroupId == NumGroups - 1);
    size_t GroupStart = GroupId * PerGroup;
    size_t GroupEnd = LastGroup ? Range.size() : (GroupStart + PerGroup);

    // Loop over the contiguous chunk
    size_t Start = GroupStart + NDId.get_local_id(0);
    size_t End = GroupEnd;
    size_t Stride = NDId.get_local_range(0);
    auto GetDelinearized = [&](size_t I) {
      auto Id = getDelinearizedId(Range, I);
      if constexpr (std::is_invocable_v<decltype(KernelFunc), id<Dims>,
                                        decltype(Reducers)...>)
        return Id;
      else
        // SYCL doesn't provide parallel_for accepting offset in presence of
        // reductions, so use with_offset==false.
        return reduction::getDelinearizedItem(Range, Id);
    };
    for (size_t I = Start; I < End; I += Stride)
      KernelFunc(GetDelinearized(I), Reducers...);
  };
  if constexpr (NumArgs == 2) {
    using Reduction = std::tuple_element_t<0, decltype(ReduTuple)>;
    auto &Redu = std::get<0>(ReduTuple);

    constexpr auto StrategyToUse = [&]() {
      if constexpr (Strategy != reduction::strategy::auto_select)
        return Strategy;

      // TODO: Both group_reduce_and_last_wg_detection and range_basic require
      // memory_order::acq_rel support that isn't guaranteed by the
      // specification. However, implementing run-time check for that would
      // result in an extra kernel compilation(s). We probably need to
      // investigate if the usage of kernel_bundles can mitigate that.
      // TODO: local_atomic_and_atomic_cross_wg uses atomics on the partial
      // results, which may add an implicit requirement on aspect::atomic64. As
      // a temporary work-around we do not pick this if the result type is
      // 64-bit. In the future this selection should be done at runtime based
      // on the device.
      // Note: Identityless reductions cannot use group reductions.
      if constexpr (Reduction::has_fast_reduce && Reduction::has_identity)
        return reduction::strategy::group_reduce_and_last_wg_detection;
      else if constexpr (Reduction::has_fast_atomics &&
                         sizeof(typename Reduction::result_type) != 8)
        return reduction::strategy::local_atomic_and_atomic_cross_wg;
      else
        return reduction::strategy::range_basic;
    }();

    reduction_parallel_for<KernelName, StrategyToUse>(CGH, NDRange, Properties,
                                                      Redu, UpdatedKernelFunc);
  } else {
    return std::apply(
        [&](auto &...Reds) {
          return reduction_parallel_for<KernelName, Strategy>(
              CGH, NDRange, Properties, Reds..., UpdatedKernelFunc);
        },
        ReduTuple);
  }
}
} // namespace detail

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction operation \p Combiner, and optional reduction properties.
/// The reduction algorithm may be less efficient if the specified binary
/// operation does not have a known identity.
template <typename T, typename AllocatorT, typename BinaryOperation>
auto reduction(buffer<T, 1, AllocatorT> Var, handler &CGH,
               BinaryOperation Combiner, const property_list &PropList = {}) {
  std::ignore = CGH;
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 0, 1, false>(
      Var, Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given USM pointer \p Var, handler \p CGH, reduction operation
/// \p Combiner, and optional reduction properties.
/// The reduction algorithm may be less efficient if the specified binary
/// operation does not have a known identity.
template <typename T, typename BinaryOperation>
auto reduction(T *Var, BinaryOperation Combiner,
               const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 0, 1, false>(
      Var, Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the given buffer \p Var, handler \p CGH,
/// reduction identity value \p Identity, reduction operation \p Combiner,
/// and optional reduction properties.
template <typename T, typename AllocatorT, typename BinaryOperation>
auto reduction(buffer<T, 1, AllocatorT> Var, handler &CGH, const T &Identity,
               BinaryOperation Combiner, const property_list &PropList = {}) {
  std::ignore = CGH;
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 0, 1, true>(
      Var, Identity, Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given USM pointer \p Var, reduction identity value \p Identity,
/// binary operation \p Combiner, and optional reduction properties.
template <typename T, typename BinaryOperation>
auto reduction(T *Var, const T &Identity, BinaryOperation Combiner,
               const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 0, 1, true>(
      Var, Identity, Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given sycl::span \p Span, reduction operation \p Combiner, and
/// optional reduction properties.
/// The reduction algorithm may be less efficient if the specified binary
/// operation does not have a known identity.
template <typename T, size_t Extent, typename BinaryOperation,
          typename = std::enable_if_t<Extent != dynamic_extent>>
auto reduction(span<T, Extent> Span, BinaryOperation Combiner,
               const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 1, Extent, false>(
      Span.data(), Combiner, InitializeToIdentity);
}

/// Constructs a reduction object using the reduction variable referenced by
/// the given sycl::span \p Span, reduction identity value \p Identity,
/// reduction operation \p Combiner, and optional reduction properties.
template <typename T, size_t Extent, typename BinaryOperation,
          typename = std::enable_if_t<Extent != dynamic_extent>>
auto reduction(span<T, Extent> Span, const T &Identity,
               BinaryOperation Combiner, const property_list &PropList = {}) {
  bool InitializeToIdentity =
      PropList.has_property<property::reduction::initialize_to_identity>();
  return detail::make_reduction<BinaryOperation, 1, Extent, true>(
      Span.data(), Identity, Combiner, InitializeToIdentity);
}
} // namespace _V1
} // namespace sycl
