//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/atomic.hpp>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/tuple.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/handler.hpp>
#include <sycl/kernel.hpp>
#include <sycl/known_identity.hpp>

#include <tuple>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace detail {

/// Base non-template class which is a base class for all reduction
/// implementation classes. It is needed to detect the reduction classes.
class reduction_impl_base {};

/// Predicate returning true if all template type parameters except the last one
/// are reductions.
template <typename FirstT, typename... RestT> struct AreAllButLastReductions {
  static constexpr bool value =
      std::is_base_of<reduction_impl_base,
                      std::remove_reference_t<FirstT>>::value &&
      AreAllButLastReductions<RestT...>::value;
};

/// Helper specialization of AreAllButLastReductions for one element only.
/// Returns true if the template parameter is not a reduction.
template <typename T> struct AreAllButLastReductions<T> {
  static constexpr bool value =
      !std::is_base_of<reduction_impl_base, std::remove_reference_t<T>>::value;
};
} // namespace detail
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#if __cplusplus >= 201703L
// Entire feature is dependent on C++17. We still have to make the trait above
// available as queue shortcuts use them unconditionally, including on
// non-reduction path.
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace detail {
template <class FunctorTy>
event withAuxHandler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost,
                     FunctorTy Func) {
  handler AuxHandler(Queue, IsHost);
  Func(AuxHandler);
  return AuxHandler.finalize();
}

using sycl::detail::bool_constant;
using sycl::detail::enable_if_t;
using sycl::detail::queue_impl;
using sycl::detail::remove_AS;

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
    bool_constant<false>;
#else
    bool_constant<((sycl::detail::is_sgenfloat<T>::value && sizeof(T) == 4) ||
                   sycl::detail::is_sgeninteger<T>::value) &&
                  sycl::detail::IsValidAtomicType<T>::value &&
                  (sycl::detail::IsPlus<T, BinaryOperation>::value ||
                   sycl::detail::IsMinimum<T, BinaryOperation>::value ||
                   sycl::detail::IsMaximum<T, BinaryOperation>::value ||
                   sycl::detail::IsBitOR<T, BinaryOperation>::value ||
                   sycl::detail::IsBitXOR<T, BinaryOperation>::value ||
                   sycl::detail::IsBitAND<T, BinaryOperation>::value)>;
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
    bool_constant<false>;
#else
    bool_constant<(sycl::detail::IsPlus<T, BinaryOperation>::value ||
                   sycl::detail::IsMinimum<T, BinaryOperation>::value ||
                   sycl::detail::IsMaximum<T, BinaryOperation>::value) &&
                  sycl::detail::is_sgenfloat<T>::value && sizeof(T) == 8>;
#endif

// This type trait is used to detect if the group algorithm reduce() used with
// operands of the type T and the operation BinaryOperation is available
// for using in reduction.
// The macro SYCL_REDUCTION_DETERMINISTIC prohibits using the reduce() algorithm
// to produce stable results across same type devices.
template <typename T, class BinaryOperation>
using IsReduOptForFastReduce =
#ifdef SYCL_REDUCTION_DETERMINISTIC
    bool_constant<false>;
#else
    bool_constant<((sycl::detail::is_sgeninteger<T>::value &&
                    (sizeof(T) == 4 || sizeof(T) == 8)) ||
                   sycl::detail::is_sgenfloat<T>::value) &&
                  (sycl::detail::IsPlus<T, BinaryOperation>::value ||
                   sycl::detail::IsMinimum<T, BinaryOperation>::value ||
                   sycl::detail::IsMaximum<T, BinaryOperation>::value)>;
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
          bool View = false, typename Subst = void>
class reducer;

/// Helper class for accessing reducer-defined types in CRTP
/// May prove to be useful for other things later
template <typename Reducer> struct ReducerTraits;

template <typename T, class BinaryOperation, int Dims, std::size_t Extent,
          bool View, typename Subst>
struct ReducerTraits<reducer<T, BinaryOperation, Dims, Extent, View, Subst>> {
  using type = T;
  using op = BinaryOperation;
  static constexpr int dims = Dims;
  static constexpr size_t extent = Extent;
};

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
  using T = typename ReducerTraits<Reducer>::type;
  using BinaryOperation = typename ReducerTraits<Reducer>::op;
  static constexpr int Dims = ReducerTraits<Reducer>::dims;
  static constexpr size_t Extent = ReducerTraits<Reducer>::extent;

public:
  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) &&
              sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++() {
    static_cast<Reducer *>(this)->combine(static_cast<T>(1));
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) &&
              sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++(int) {
    static_cast<Reducer *>(this)->combine(static_cast<T>(1));
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) && sycl::detail::IsPlus<_T, BinaryOperation>::value>
  operator+=(const _T &Partial) {
    static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) &&
              sycl::detail::IsMultiplies<_T, BinaryOperation>::value>
  operator*=(const _T &Partial) {
    static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) && sycl::detail::IsBitOR<_T, BinaryOperation>::value>
  operator|=(const _T &Partial) {
    static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) &&
              sycl::detail::IsBitXOR<_T, BinaryOperation>::value>
  operator^=(const _T &Partial) {
    static_cast<Reducer *>(this)->combine(Partial);
  }

  template <typename _T = T, int _Dims = Dims>
  enable_if_t<(_Dims == 0) &&
              sycl::detail::IsBitAND<_T, BinaryOperation>::value>
  operator&=(const _T &Partial) {
    static_cast<Reducer *>(this)->combine(Partial);
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
      auto AtomicRef =
          sycl::atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(),
                           Space>(multi_ptr<T, Space>(ReduVarPtr)[E]);
      Functor(AtomicRef, reducer->getElement(E));
    }
  }

  template <class _T, access::address_space Space, class BinaryOperation>
  static constexpr bool BasicCheck =
      std::is_same<typename remove_AS<_T>::type, T>::value &&
      (Space == access::address_space::global_space ||
       Space == access::address_space::local_space);

public:
  /// Atomic ADD operation: *ReduVarPtr += MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
              (IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value ||
               IsReduOptForAtomic64Op<T, _BinaryOperation>::value) &&
              sycl::detail::IsPlus<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_add(Val); });
  }

  /// Atomic BITWISE OR operation: *ReduVarPtr |= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsBitOR<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_or(Val); });
  }

  /// Atomic BITWISE XOR operation: *ReduVarPtr ^= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsBitXOR<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_xor(Val); });
  }

  /// Atomic BITWISE AND operation: *ReduVarPtr &= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsBitAND<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_and(Val); });
  }

  /// Atomic MIN operation: *ReduVarPtr = sycl::minimum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
              (IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value ||
               IsReduOptForAtomic64Op<T, _BinaryOperation>::value) &&
              sycl::detail::IsMinimum<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_min(Val); });
  }

  /// Atomic MAX operation: *ReduVarPtr = sycl::maximum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<BasicCheck<_T, Space, _BinaryOperation> &&
              (IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value ||
               IsReduOptForAtomic64Op<T, _BinaryOperation>::value) &&
              sycl::detail::IsMaximum<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_combine_impl<Space>(
        ReduVarPtr, [](auto Ref, auto Val) { return Ref.fetch_max(Val); });
  }
};

/// Specialization of the generic class 'reducer'. It is used for reductions
/// of those types and operations for which the identity value is not known.
///
/// It stores a copy of the identity and binary operation associated with the
/// reduction.
template <typename T, class BinaryOperation, int Dims, size_t Extent, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, View,
    enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                !sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public combiner<
          reducer<T, BinaryOperation, Dims, Extent, View,
                  enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                              !sycl::detail::IsKnownIdentityOp<
                                  T, BinaryOperation>::value>>> {
public:
  reducer(const T &Identity, BinaryOperation BOp)
      : MValue(Identity), MIdentity(Identity), MBinaryOp(BOp) {}

  void combine(const T &Partial) { MValue = MBinaryOp(MValue, Partial); }

  T getIdentity() const { return MIdentity; }

  T &getElement(size_t) { return MValue; }
  const T &getElement(size_t) const { return MValue; }
  T MValue;

private:
  const T MIdentity;
  BinaryOperation MBinaryOp;
};

/// Specialization of the generic class 'reducer'. It is used for reductions
/// of those types and operations for which the identity value is known.
///
/// It allows to reduce the size of the 'reducer' object by not holding
/// the identity field inside it and allows to add a default constructor.
template <typename T, class BinaryOperation, int Dims, size_t Extent, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, View,
    enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public combiner<
          reducer<T, BinaryOperation, Dims, Extent, View,
                  enable_if_t<Dims == 0 && Extent == 1 && View == false &&
                              sycl::detail::IsKnownIdentityOp<
                                  T, BinaryOperation>::value>>> {
public:
  reducer() : MValue(getIdentity()) {}
  reducer(const T & /* Identity */, BinaryOperation) : MValue(getIdentity()) {}

  void combine(const T &Partial) {
    BinaryOperation BOp;
    MValue = BOp(MValue, Partial);
  }

  static T getIdentity() {
    return sycl::detail::known_identity_impl<BinaryOperation, T>::value;
  }

  T &getElement(size_t) { return MValue; }
  const T &getElement(size_t) const { return MValue; }
  T MValue;
};

/// Component of 'reducer' class for array reductions, representing a single
/// element of the span (as returned by the subscript operator).
template <typename T, class BinaryOperation, int Dims, size_t Extent, bool View>
class reducer<T, BinaryOperation, Dims, Extent, View,
              enable_if_t<Dims == 0 && View == true>>
    : public combiner<reducer<T, BinaryOperation, Dims, Extent, View,
                              enable_if_t<Dims == 0 && View == true>>> {
public:
  reducer(T &Ref, BinaryOperation BOp) : MElement(Ref), MBinaryOp(BOp) {}

  void combine(const T &Partial) { MElement = MBinaryOp(MElement, Partial); }

private:
  T &MElement;
  BinaryOperation MBinaryOp;
};

/// Specialization of 'reducer' class for array reductions exposing the
/// subscript operator.
template <typename T, class BinaryOperation, int Dims, size_t Extent, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, View,
    enable_if_t<Dims == 1 && View == false &&
                !sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public combiner<reducer<T, BinaryOperation, Dims, Extent, View,
                              enable_if_t<Dims == 1 && View == false &&
                                          !sycl::detail::IsKnownIdentityOp<
                                              T, BinaryOperation>::value>>> {
public:
  reducer(const T &Identity, BinaryOperation BOp)
      : MValue(Identity), MIdentity(Identity), MBinaryOp(BOp) {}

  reducer<T, BinaryOperation, Dims - 1, Extent, true> operator[](size_t Index) {
    return {MValue[Index], MBinaryOp};
  }

  T getIdentity() const { return MIdentity; }
  T &getElement(size_t E) { return MValue[E]; }
  const T &getElement(size_t E) const { return MValue[E]; }

private:
  marray<T, Extent> MValue;
  const T MIdentity;
  BinaryOperation MBinaryOp;
};

/// Specialization of 'reducer' class for array reductions accepting a span
/// in cases where the identity value is known.
template <typename T, class BinaryOperation, int Dims, size_t Extent, bool View>
class reducer<
    T, BinaryOperation, Dims, Extent, View,
    enable_if_t<Dims == 1 && View == false &&
                sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
    : public combiner<reducer<T, BinaryOperation, Dims, Extent, View,
                              enable_if_t<Dims == 1 && View == false &&
                                          sycl::detail::IsKnownIdentityOp<
                                              T, BinaryOperation>::value>>> {
public:
  reducer() : MValue(getIdentity()) {}
  reducer(const T & /* Identity */, BinaryOperation) : MValue(getIdentity()) {}

  // SYCL 2020 revision 4 says this should be const, but this is a bug
  // see https://github.com/KhronosGroup/SYCL-Docs/pull/252
  reducer<T, BinaryOperation, Dims - 1, Extent, true> operator[](size_t Index) {
    return {MValue[Index], BinaryOperation()};
  }

  static T getIdentity() {
    return sycl::detail::known_identity_impl<BinaryOperation, T>::value;
  }

  T &getElement(size_t E) { return MValue[E]; }
  const T &getElement(size_t E) const { return MValue[E]; }

private:
  marray<T, Extent> MValue;
};

/// Templated class for common functionality of all reduction implementation
/// classes.
template <typename T, class BinaryOperation> class reduction_impl_common {
protected:
  reduction_impl_common(const T &Identity, BinaryOperation BinaryOp,
                        bool Init = false)
      : MIdentity(Identity), MBinaryOp(BinaryOp), InitializeToIdentity(Init) {}

public:
  /// Returns the statically known identity value.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<sycl::detail::IsKnownIdentityOp<_T, _BinaryOperation>::value,
              _T> constexpr getIdentity() {
    return sycl::detail::known_identity_impl<_BinaryOperation, _T>::value;
  }

  /// Returns the identity value given by user.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<!sycl::detail::IsKnownIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return MIdentity;
  }

  /// Returns the binary operation associated with the reduction.
  BinaryOperation getBinaryOperation() const { return MBinaryOp; }
  bool initializeToIdentity() const { return InitializeToIdentity; }

protected:
  /// Identity of the BinaryOperation.
  /// The result of BinaryOperation(X, MIdentity) is equal to X for any X.
  const T MIdentity;

  BinaryOperation MBinaryOp;
  bool InitializeToIdentity;
};

template <class T> struct is_rw_acc_t : public std::false_type {};

template <class T, int AccessorDims, access::placeholder IsPlaceholder,
          typename PropList>
struct is_rw_acc_t<accessor<T, AccessorDims, access::mode::read_write,
                            access::target::device, IsPlaceholder, PropList>>
    : public std::true_type {};

template <class T> struct is_dw_acc_t : public std::false_type {};

template <class T, int AccessorDims, access::placeholder IsPlaceholder,
          typename PropList>
struct is_dw_acc_t<accessor<T, AccessorDims, access::mode::discard_write,
                            access::target::device, IsPlaceholder, PropList>>
    : public std::true_type {};

template <class T> struct is_placeholder_t : public std::false_type {};

template <class T, int AccessorDims, access::mode Mode, typename PropList>
struct is_placeholder_t<accessor<T, AccessorDims, Mode, access::target::device,
                                 access::placeholder::true_t, PropList>>
    : public std::true_type {};

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

template <class T, int AccessorDims, access::mode Mode,
          access::placeholder IsPH, typename PropList>
struct get_red_t<
    accessor<T, AccessorDims, Mode, access::target::device, IsPH, PropList>> {
  using type = T;
};

template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename RedOutVar>
class reduction_impl_algo : public reduction_impl_common<T, BinaryOperation> {
  using base = reduction_impl_common<T, BinaryOperation>;
  using self = reduction_impl_algo<T, BinaryOperation, Dims, Extent, RedOutVar>;

public:
  using reducer_type = reducer<T, BinaryOperation, Dims, Extent>;
  using result_type = T;
  using binary_operation = BinaryOperation;

  // Buffers and accessors always describe scalar reductions (i.e. Dims == 0)
  // The input buffer/accessor is allowed to have different dimensionality
  // AccessorDims also determines the dimensionality of some temp storage
  static constexpr int accessor_dim = data_dim_t<RedOutVar>::value;
  static constexpr int buffer_dim = (accessor_dim == 0) ? 1 : accessor_dim;
  static constexpr access::placeholder is_placeholder =
      is_placeholder_t<RedOutVar>::value ? access::placeholder::true_t
                                         : access::placeholder::false_t;
  using rw_accessor_type = accessor<T, accessor_dim, access::mode::read_write,
                                    access::target::device, is_placeholder,
                                    ext::oneapi::accessor_property_list<>>;
  static constexpr bool has_float64_atomics =
      IsReduOptForAtomic64Op<T, BinaryOperation>::value;
  static constexpr bool has_fast_atomics =
      IsReduOptForFastAtomicFetch<T, BinaryOperation>::value;
  static constexpr bool has_fast_reduce =
      IsReduOptForFastReduce<T, BinaryOperation>::value;

  static constexpr bool is_usm = std::is_same_v<RedOutVar, T *>;

  static constexpr bool is_rw_acc = is_rw_acc_t<RedOutVar>::value;
  static constexpr bool is_dw_acc = is_dw_acc_t<RedOutVar>::value;
  static constexpr bool is_acc = is_rw_acc | is_dw_acc;
  static_assert(!is_rw_acc || !is_dw_acc, "Can be only one at once!");
  static_assert(!is_usm || !is_acc, "Can be only one at once!");

  static constexpr size_t dims = Dims;
  static constexpr size_t num_elements = Extent;

  reduction_impl_algo(const T &Identity, BinaryOperation BinaryOp, bool Init,
                      RedOutVar RedOut)
      : base(Identity, BinaryOp, Init), MRedOut(std::move(RedOut)){};

  /// Creates and returns a local accessor with the \p Size elements.
  /// By default the local accessor elements are of the same type as the
  /// elements processed by the reduction, but may it be altered by specifying
  /// \p _T explicitly if need an accessor with elements of different type.
  template <typename _T = result_type>
  static accessor<_T, buffer_dim, access::mode::read_write,
                  access::target::local>
  getReadWriteLocalAcc(size_t Size, handler &CGH) {
    return {Size, CGH};
  }

  accessor<T, buffer_dim, access::mode::read>
  getReadAccToPreviousPartialReds(handler &CGH) const {
    CGH.addReduction(MOutBufPtr);
    return {*MOutBufPtr, CGH};
  }

  template <bool IsOneWG>
  auto getWriteMemForPartialReds(size_t Size, handler &CGH) {
    // If there is only one WG we can avoid creation of temporary buffer with
    // partial sums and write directly into user's reduction variable.
    //
    // Current implementation doesn't allow that in case of DW accessor used for
    // reduction because C++ types for it and for temporary storage don't match,
    // hence the second part of the check.
    if constexpr (IsOneWG && !is_dw_acc) {
      return MRedOut;
    } else {
      MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
      CGH.addReduction(MOutBufPtr);
      return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
    }
  }

  template <class _T = T, int D = buffer_dim>
  auto &getTempBuffer(size_t Size, handler &CGH) {
    auto Buffer = std::make_shared<buffer<_T, D>>(range<1>(Size));
    CGH.addReduction(Buffer);
    return *Buffer;
  }

  /// Returns an accessor accessing the memory that will hold the reduction
  /// partial sums.
  /// If \p Size is equal to one, then the reduction result is the final and
  /// needs to be written to user's read-write accessor (if there is such).
  /// Otherwise, a new buffer is created and accessor to that buffer is
  /// returned.
  rw_accessor_type getWriteAccForPartialReds(size_t Size, handler &CGH) {
    if constexpr (is_rw_acc) {
      if (Size == 1) {
        CGH.associateWithHandler(&MRedOut, access::target::device);
        return MRedOut;
      }
    }

    // Create a new output buffer and return an accessor to it.
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
  }

  /// If reduction is initialized with read-write accessor, which does not
  /// require initialization with identity value, then return user's read-write
  /// accessor. Otherwise, create global buffer with 'num_elements' initialized
  /// with identity value and return an accessor to that buffer.
  template <bool HasFastAtomics = (has_fast_atomics || has_float64_atomics)>
  std::enable_if_t<HasFastAtomics, rw_accessor_type>
  getReadWriteAccessorToInitializedMem(handler &CGH) {
    if constexpr (is_rw_acc) {
      if (!base::initializeToIdentity())
        return MRedOut;
    }
    assert(!(is_dw_acc && !base::initializeToIdentity()) &&
           "Unexpected condition!");

    // TODO: Move to T[] in C++20 to simplify handling here
    // auto RWReduVal = std::make_shared<T[num_elements]>();
    auto RWReduVal = std::make_shared<std::array<T, num_elements>>();
    for (int i = 0; i < num_elements; ++i) {
      (*RWReduVal)[i] = base::getIdentity();
    }
    CGH.addReduction(RWReduVal);
    MOutBufPtr = std::make_shared<buffer<T, 1>>(RWReduVal.get()->data(),
                                                range<1>(num_elements));
    MOutBufPtr->set_final_data();
    CGH.addReduction(MOutBufPtr);
    return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
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
  template <typename Name> auto getGroupsCounterAccDiscrete(handler &CGH) {
    auto &Buf = getTempBuffer<int, 1>(1, CGH);
    std::shared_ptr<detail::queue_impl> QueueCopy = CGH.MQueue;
    auto Event = CGH.withAuxHandler(QueueCopy, [&](handler &InitHandler) {
      auto Acc = accessor{Buf, InitHandler, sycl::write_only, sycl::no_init};
      InitHandler.single_task<Name>([=]() { Acc[0] = 0; });
    });
    CGH.depends_on(Event);
    return accessor{Buf, CGH};
  }

  RedOutVar &getUserRedVar() { return MRedOut; }

  static inline result_type *getOutPointer(result_type *OutPtr) {
    return OutPtr;
  }
  template <class AccessorType>
  static inline result_type *getOutPointer(const AccessorType &OutAcc) {
    return OutAcc.get_pointer().get();
  }

private:
  template <typename BufferT>
  rw_accessor_type createHandlerWiredReadWriteAccessor(handler &CGH,
                                                       BufferT Buffer) {
    // TODO:
    // SYCL 2020: The accessor template parameter IsPlaceholder is allowed to be
    // specified, but it has no bearing on whether the accessor instance is a
    // placeholder. This is determined solely by the constructor used to create
    // the instance. The associated type access::placeholder is also deprecated.
    if constexpr (is_placeholder == access::placeholder::true_t) {
      rw_accessor_type Acc(Buffer);
      CGH.require(Acc);
      return Acc;
    } else {
      return {Buffer, CGH};
    }
  }

  std::shared_ptr<buffer<T, buffer_dim>> MOutBufPtr;

  /// User's accessor/USM pointer to where the reduction must be written.
  RedOutVar MRedOut;
};
/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename RedOutVar>
class reduction_impl
    : private reduction_impl_base,
      public reduction_impl_algo<T, BinaryOperation, Dims, Extent, RedOutVar> {
private:
  using algo = reduction_impl_algo<T, BinaryOperation, Dims, Extent, RedOutVar>;
  using self = reduction_impl<T, BinaryOperation, Dims, Extent, RedOutVar>;

  static constexpr bool is_known_identity =
      sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value;

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
      return reducer_type::getIdentity();

    } else {
      return Identity;
    }
  }

public:
  using algo::is_acc;
  using algo::is_dw_acc;
  using algo::is_rw_acc;
  using algo::is_usm;

  using reducer_type = typename algo::reducer_type;
  using rw_accessor_type = typename algo::rw_accessor_type;

  // Only scalar and 1D array reductions are supported by SYCL 2020.
  static_assert(Dims <= 1, "Multi-dimensional reductions are not supported.");

  /// Constructs reduction_impl when the identity value is statically known.
  template <typename _self = self,
            enable_if_t<_self::is_known_identity && _self::is_acc> * = nullptr>
  reduction_impl(RedOutVar &Acc)
      : algo(reducer_type::getIdentity(), BinaryOperation(), is_dw_acc, Acc) {
    if (Acc.size() != 1)
      throw sycl::runtime_error(errc::invalid,
                                "Reduction variable must be a scalar.",
                                PI_ERROR_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is statically known.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <typename _self = self,
            enable_if_t<_self::is_known_identity && _self::is_usm> * = nullptr>
  reduction_impl(RedOutVar VarPtr, bool InitializeToIdentity = false)
      : algo(reducer_type::getIdentity(), BinaryOperation(),
             InitializeToIdentity, VarPtr) {}

  /// SYCL-2020.
  /// Constructs reduction_impl when the identity value is statically known.
  template <typename _self = self,
            std::enable_if_t<_self::is_known_identity && _self::is_rw_acc> * =
                nullptr>
  reduction_impl(RedOutVar &Acc, handler &CGH, bool InitializeToIdentity)
      : algo(reducer_type::getIdentity(), BinaryOperation(),
             InitializeToIdentity, Acc) {
    associateWithHandler(CGH, &Acc, access::target::device);
    if (Acc.size() != 1)
      throw sycl::runtime_error(errc::invalid,
                                "Reduction variable must be a scalar.",
                                PI_ERROR_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is unknown.
  template <typename _self = self, enable_if_t<_self::is_acc> * = nullptr>
  reduction_impl(RedOutVar &Acc, const T &Identity, BinaryOperation BOp)
      : algo(chooseIdentity(Identity), BOp, is_dw_acc, Acc) {
    if (Acc.size() != 1)
      throw sycl::runtime_error(errc::invalid,
                                "Reduction variable must be a scalar.",
                                PI_ERROR_INVALID_VALUE);
  }

  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <typename _self = self, enable_if_t<_self::is_usm> * = nullptr>
  reduction_impl(RedOutVar VarPtr, const T &Identity, BinaryOperation BOp,
                 bool InitializeToIdentity = false)
      : algo(chooseIdentity(Identity), BOp, InitializeToIdentity, VarPtr) {}

  /// For placeholder accessor.
  template <typename _self = self, enable_if_t<_self::is_rw_acc> * = nullptr>
  reduction_impl(RedOutVar &Acc, handler &CGH, const T &Identity,
                 BinaryOperation BOp, bool InitializeToIdentity)
      : algo(chooseIdentity(Identity), BOp, InitializeToIdentity, Acc) {
    associateWithHandler(CGH, &Acc, access::target::device);
    if (Acc.size() != 1)
      throw sycl::runtime_error(errc::invalid,
                                "Reduction variable must be a scalar.",
                                PI_ERROR_INVALID_VALUE);
  }
};

template <class BinaryOp, int Dims, size_t Extent, typename RedOutVar,
          typename... RestTy>
auto make_reduction(RedOutVar RedVar, RestTy &&...Rest) {
  return reduction_impl<typename get_red_t<RedOutVar>::type, BinaryOp, Dims,
                        Extent, RedOutVar>{RedVar,
                                           std::forward<RestTy>(Rest)...};
}

/// A helper to pass undefined (sycl::detail::auto_name) names unmodified. We
/// must do that to avoid name collisions.
template <template <typename...> class Namer, class KernelName, class... Ts>
using __sycl_reduction_kernel =
    std::conditional_t<std::is_same<KernelName, sycl::detail::auto_name>::value,
                       sycl::detail::auto_name, Namer<KernelName, Ts...>>;

/// Called in device code. This function iterates through the index space
/// \p Range using stride equal to the global range specified in \p NdId,
/// which gives much better performance than using stride equal to 1.
/// For each of the index the given \p F function/functor is called and
/// the reduction value hold in \p Reducer is accumulated in those calls.
template <typename KernelFunc, int Dims, typename ReducerT>
void reductionLoop(const range<Dims> &Range, ReducerT &Reducer,
                   const nd_item<1> &NdId, KernelFunc &F) {
  size_t Start = NdId.get_global_id(0);
  size_t End = Range.size();
  size_t Stride = NdId.get_global_range(0);
  for (size_t I = Start; I < End; I += Stride)
    F(sycl::detail::getDelinearizedId(Range, I), Reducer);
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct RangeFastAtomics;
} // namespace main_krn
} // namespace reduction
template <typename KernelName, typename KernelType, int Dims, class Reduction>
bool reduCGFuncForRangeFastAtomics(handler &CGH, KernelType KernelFunc,
                                   const range<Dims> &Range,
                                   const nd_range<1> &NDRange,
                                   Reduction &Redu) {
  constexpr size_t NElements = Reduction::num_elements;
  auto Out = Redu.getReadWriteAccessorToInitializedMem(CGH);
  auto GroupSum = Reduction::getReadWriteLocalAcc(NElements, CGH);
  using Name = __sycl_reduction_kernel<reduction::main_krn::RangeFastAtomics,
                                       KernelName>;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    // Work-group cooperates to initialize multiple reduction variables
    auto LID = NDId.get_local_id(0);
    for (size_t E = LID; E < NElements; E += NDId.get_local_range(0)) {
      GroupSum[E] = Reducer.getIdentity();
    }
    sycl::detail::workGroupBarrier();

    // Each work-item has its own reducer to combine
    Reducer.template atomic_combine<access::address_space::local_space>(
        &GroupSum[0]);

    // Single work-item performs finalization for entire work-group
    // TODO: Opportunity to parallelize across elements
    sycl::detail::workGroupBarrier();
    if (LID == 0) {
      for (size_t E = 0; E < NElements; ++E) {
        Reducer.getElement(E) = GroupSum[E];
      }
      Reducer.template atomic_combine(Reduction::getOutPointer(Out));
    }
  });
  return Reduction::is_usm || Redu.initializeToIdentity();
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct RangeFastReduce;
} // namespace main_krn
namespace init_krn {
template <class KernelName> struct GroupCounter;
}
} // namespace reduction
template <typename KernelName, typename KernelType, int Dims, class Reduction>
bool reduCGFuncForRangeFastReduce(handler &CGH, KernelType KernelFunc,
                                  const range<Dims> &Range,
                                  const nd_range<1> &NDRange, Reduction &Redu) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t WGSize = NDRange.get_local_range().size();
  size_t NWorkGroups = NDRange.get_group_range().size();

  auto &Out = Redu.getUserRedVar();
  if constexpr (Reduction::is_acc)
    associateWithHandler(CGH, &Out, access::target::device);

  auto &PartialSumsBuf = Redu.getTempBuffer(NWorkGroups * NElements, CGH);
  accessor PartialSums(PartialSumsBuf, CGH, sycl::read_write, sycl::no_init);

  bool IsUpdateOfUserVar = !Reduction::is_usm && !Redu.initializeToIdentity();
  using InitName =
      __sycl_reduction_kernel<reduction::init_krn::GroupCounter, KernelName>;

  // Integrated/discrete GPUs have different faster path.
  auto NWorkGroupsFinished =
      sycl::detail::getDeviceFromHandler(CGH)
              .get_info<info::device::host_unified_memory>()
          ? Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH)
          : Redu.template getGroupsCounterAccDiscrete<InitName>(CGH);

  auto DoReducePartialSumsInLastWG =
      Reduction::template getReadWriteLocalAcc<int>(1, CGH);

  using Name =
      __sycl_reduction_kernel<reduction::main_krn::RangeFastReduce, KernelName>;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    typename Reduction::binary_operation BOp;
    auto Group = NDId.get_group();

    // If there are multiple values, reduce each separately
    // reduce_over_group is only defined for each T, not for span<T, ...>
    size_t LID = NDId.get_local_id(0);
    for (int E = 0; E < NElements; ++E) {
      auto &RedElem = Reducer.getElement(E);
      RedElem = reduce_over_group(Group, RedElem, BOp);
      if (LID == 0) {
        if (NWorkGroups == 1) {
          auto &OutElem = Reduction::getOutPointer(Out)[E];
          // Can avoid using partial sum and write the final result immediately.
          if (IsUpdateOfUserVar)
            RedElem = BOp(RedElem, OutElem);
          OutElem = RedElem;
        } else {
          PartialSums[NDId.get_group_linear_id() * NElements + E] =
              Reducer.getElement(E);
        }
      }
    }

    if (NWorkGroups == 1)
      // We're done.
      return;

    // Signal this work-group has finished after all values are reduced
    if (LID == 0) {
      auto NFinished =
          sycl::atomic_ref<int, memory_order::relaxed, memory_scope::device,
                           access::address_space::global_space>(
              NWorkGroupsFinished[0]);
      DoReducePartialSumsInLastWG[0] = ++NFinished == NWorkGroups;
    }

    sycl::detail::workGroupBarrier();
    if (DoReducePartialSumsInLastWG[0]) {
      // Reduce each result separately
      // TODO: Opportunity to parallelize across elements.
      for (int E = 0; E < NElements; ++E) {
        auto &OutElem = Reduction::getOutPointer(Out)[E];
        auto LocalSum = Reducer.getIdentity();
        for (size_t I = LID; I < NWorkGroups; I += WGSize)
          LocalSum = BOp(LocalSum, PartialSums[I * NElements + E]);
        auto Result = reduce_over_group(Group, LocalSum, BOp);

        if (LID == 0) {
          if (IsUpdateOfUserVar)
            Result = BOp(Result, OutElem);
          OutElem = Result;
        }
      }
    }
  });

  // We've updated user's variable, no extra work needed.
  return false;
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct RangeBasic;
} // namespace main_krn
} // namespace reduction
template <typename KernelName, typename KernelType, int Dims, class Reduction>
bool reduCGFuncForRangeBasic(handler &CGH, KernelType KernelFunc,
                             const range<Dims> &Range,
                             const nd_range<1> &NDRange, Reduction &Redu) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t WGSize = NDRange.get_local_range().size();
  size_t NWorkGroups = NDRange.get_group_range().size();

  bool IsUpdateOfUserVar = !Reduction::is_usm && !Redu.initializeToIdentity();
  auto PartialSums =
      Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);
  auto Out = (NWorkGroups == 1)
                 ? PartialSums
                 : Redu.getWriteAccForPartialReds(NElements, CGH);
  auto LocalReds = Reduction::getReadWriteLocalAcc(WGSize + 1, CGH);
  auto NWorkGroupsFinished =
      Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH);
  auto DoReducePartialSumsInLastWG =
      Reduction::template getReadWriteLocalAcc<int>(1, CGH);

  auto Identity = Redu.getIdentity();
  auto BOp = Redu.getBinaryOperation();
  using Name =
      __sycl_reduction_kernel<reduction::main_krn::RangeBasic, KernelName>;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer(Identity, BOp);
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    // If there are multiple values, reduce each separately
    // This prevents local memory from scaling with elements
    size_t LID = NDId.get_local_linear_id();
    for (int E = 0; E < NElements; ++E) {

      // Copy the element to local memory to prepare it for tree-reduction.
      LocalReds[LID] = Reducer.getElement(E);
      if (LID == 0)
        LocalReds[WGSize] = Identity;
      sycl::detail::workGroupBarrier();

      // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0].
      // LocalReds[WGSize] accumulates last/odd elements when the step
      // of tree-reduction loop is not even.
      size_t PrevStep = WGSize;
      for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
        if (LID < CurStep)
          LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
        else if (LID == CurStep && (PrevStep & 0x1))
          LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
        sycl::detail::workGroupBarrier();
        PrevStep = CurStep;
      }

      if (LID == 0) {
        auto V = BOp(LocalReds[0], LocalReds[WGSize]);
        if (NWorkGroups == 1 && IsUpdateOfUserVar)
          V = BOp(V, Reduction::getOutPointer(Out)[E]);
        // if NWorkGroups == 1, then PartialsSum and Out point to same memory.
        Reduction::getOutPointer(
            PartialSums)[NDId.get_group_linear_id() * NElements + E] = V;
      }
    }

    // Signal this work-group has finished after all values are reduced
    if (LID == 0) {
      auto NFinished =
          sycl::atomic_ref<int, memory_order::relaxed, memory_scope::device,
                           access::address_space::global_space>(
              NWorkGroupsFinished[0]);
      DoReducePartialSumsInLastWG[0] =
          ++NFinished == NWorkGroups && NWorkGroups > 1;
    }

    sycl::detail::workGroupBarrier();
    if (DoReducePartialSumsInLastWG[0]) {
      // Reduce each result separately
      // TODO: Opportunity to parallelize across elements
      for (int E = 0; E < NElements; ++E) {
        auto LocalSum = Identity;
        for (size_t I = LID; I < NWorkGroups; I += WGSize)
          LocalSum =
              BOp(LocalSum,
                  Reduction::getOutPointer(PartialSums)[I * NElements + E]);

        LocalReds[LID] = LocalSum;
        if (LID == 0)
          LocalReds[WGSize] = Identity;
        sycl::detail::workGroupBarrier();

        size_t PrevStep = WGSize;
        for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
          if (LID < CurStep)
            LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
          else if (LID == CurStep && (PrevStep & 0x1))
            LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
          sycl::detail::workGroupBarrier();
          PrevStep = CurStep;
        }
        if (LID == 0) {
          auto V = BOp(LocalReds[0], LocalReds[WGSize]);
          if (IsUpdateOfUserVar)
            V = BOp(V, Reduction::getOutPointer(Out)[E]);
          Reduction::getOutPointer(Out)[E] = V;
        }
      }
    }
  });
  return Reduction::is_usm || Reduction::is_dw_acc;
}

/// Returns "true" if the result has to be saved to user's variable by
/// reduSaveFinalResultToUserMem.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
bool reduCGFuncForRange(handler &CGH, KernelType KernelFunc,
                        const range<Dims> &Range, size_t MaxWGSize,
                        uint32_t NumConcurrentWorkGroups, Reduction &Redu) {
  size_t NWorkItems = Range.size();
  size_t WGSize = std::min(NWorkItems, MaxWGSize);
  size_t NWorkGroups = NWorkItems / WGSize;
  if (NWorkItems % WGSize)
    NWorkGroups++;
  size_t MaxNWorkGroups = NumConcurrentWorkGroups;
  NWorkGroups = std::min(NWorkGroups, MaxNWorkGroups);
  size_t NDRItems = NWorkGroups * WGSize;
  nd_range<1> NDRange{range<1>{NDRItems}, range<1>{WGSize}};

  if constexpr (Reduction::has_fast_atomics)
    return reduCGFuncForRangeFastAtomics<KernelName>(CGH, KernelFunc, Range,
                                                     NDRange, Redu);
  else if constexpr (Reduction::has_fast_reduce)
    return reduCGFuncForRangeFastReduce<KernelName>(CGH, KernelFunc, Range,
                                                    NDRange, Redu);
  else
    return reduCGFuncForRangeBasic<KernelName>(CGH, KernelFunc, Range, NDRange,
                                               Redu);
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct NDRangeBothFastReduceAndAtomics;
} // namespace main_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups, then it calls fast SYCL atomic operations to update
/// the given reduction variable \p Out.
///
/// Briefly: calls user's lambda, ext::oneapi::reduce() + atomic, INT +
/// ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFuncForNDRangeBothFastReduceAndAtomics(
    handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
    Reduction &, typename Reduction::rw_accessor_type Out) {
  constexpr size_t NElements = Reduction::num_elements;
  using Name = __sycl_reduction_kernel<
      reduction::main_krn::NDRangeBothFastReduceAndAtomics, KernelName>;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's function. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    typename Reduction::binary_operation BOp;
    for (int E = 0; E < NElements; ++E) {
      Reducer.getElement(E) =
          reduce_over_group(NDIt.get_group(), Reducer.getElement(E), BOp);
    }
    if (NDIt.get_local_linear_id() == 0)
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
  });
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct NDRangeFastAtomicsOnly;
} // namespace main_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups, then it calls fast SYCL atomic operations to update
/// user's reduction variable.
///
/// Briefly: calls user's lambda, tree-reduction + atomic, INT + AND/OR/XOR.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFuncForNDRangeFastAtomicsOnly(
    handler &CGH, bool IsPow2WG, KernelType KernelFunc,
    const nd_range<Dims> &Range, Reduction &,
    typename Reduction::rw_accessor_type Out) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t WGSize = Range.get_local_range().size();

  // Use local memory to reduce elements in work-groups into zero-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch reduce elements that could
  // otherwise be lost in the tree-reduction algorithm used in the kernel.
  size_t NLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NLocalElements, CGH);

  using Name =
      __sycl_reduction_kernel<reduction::main_krn::NDRangeFastAtomicsOnly,
                              KernelName>;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();

    // If there are multiple values, reduce each separately
    // This prevents local memory from scaling with elements
    for (int E = 0; E < NElements; ++E) {

      // Copy the element to local memory to prepare it for tree-reduction.
      LocalReds[LID] = Reducer.getElement(E);
      if (!IsPow2WG)
        LocalReds[WGSize] = Reducer.getIdentity();
      NDIt.barrier();

      // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0].
      // LocalReds[WGSize] accumulates last/odd elements when the step
      // of tree-reduction loop is not even.
      typename Reduction::binary_operation BOp;
      size_t PrevStep = WGSize;
      for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
        if (LID < CurStep)
          LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
        else if (!IsPow2WG && LID == CurStep && (PrevStep & 0x1))
          LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
        NDIt.barrier();
        PrevStep = CurStep;
      }

      if (LID == 0) {
        Reducer.getElement(E) =
            IsPow2WG ? LocalReds[0] : BOp(LocalReds[0], LocalReds[WGSize]);
      }

      // Ensure item 0 is finished with LocalReds before next iteration
      if (E != NElements - 1) {
        NDIt.barrier();
      }
    }

    if (LID == 0) {
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
    }
  });
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct NDRangeFastReduceOnly;
} // namespace main_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that
/// calls user's lambda function and does one iteration of reduction
/// of elements in each of work-groups.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: user's lambda, ext::oneapi::reduce(), FP + ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFuncForNDRangeFastReduceOnly(
    handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
    Reduction &Redu, typename Reduction::rw_accessor_type Out) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t NWorkGroups = Range.get_group_range().size();
  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;

  using Name =
      __sycl_reduction_kernel<reduction::main_krn::NDRangeFastReduceOnly,
                              KernelName>;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    // Compute the partial sum/reduction for the work-group.
    size_t WGID = NDIt.get_group_linear_id();
    typename Reduction::binary_operation BOp;
    for (int E = 0; E < NElements; ++E) {
      typename Reduction::result_type PSum;
      PSum = Reducer.getElement(E);
      PSum = reduce_over_group(NDIt.get_group(), PSum, BOp);
      if (NDIt.get_local_linear_id() == 0) {
        if (IsUpdateOfUserVar)
          PSum = BOp(Reduction::getOutPointer(Out)[E], PSum);
        Reduction::getOutPointer(Out)[WGID * NElements + E] = PSum;
      }
    }
  });
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct NDRangeBasic;
} // namespace main_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function \param KernelFunc and does one iteration of
/// reduction of elements in each of work-groups.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups. At the end of each work-group the partial sum is written
/// to a global buffer.
///
/// Briefly: user's lambda, tree-reduction, CUSTOM types/ops.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFuncForNDRangeBasic(handler &CGH, bool IsPow2WG,
                               KernelType KernelFunc,
                               const nd_range<Dims> &Range, Reduction &Redu,
                               typename Reduction::rw_accessor_type Out) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t WGSize = Range.get_local_range().size();
  size_t NWorkGroups = Range.get_group_range().size();

  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;

  // Use local memory to reduce elements in work-groups into 0-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch elements that could
  // otherwise be lost in the tree-reduction algorithm.
  size_t NumLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NumLocalElements, CGH);
  typename Reduction::result_type ReduIdentity = Redu.getIdentity();
  using Name =
      __sycl_reduction_kernel<reduction::main_krn::NDRangeBasic, KernelName>;
  auto BOp = Redu.getBinaryOperation();
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer(ReduIdentity, BOp);
    KernelFunc(NDIt, Reducer);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();

    // If there are multiple values, reduce each separately
    // This prevents local memory from scaling with elements
    for (int E = 0; E < NElements; ++E) {

      // Copy the element to local memory to prepare it for tree-reduction.
      LocalReds[LID] = Reducer.getElement(E);
      if (!IsPow2WG)
        LocalReds[WGSize] = ReduIdentity;
      NDIt.barrier();

      // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
      // LocalReds[WGSize] accumulates last/odd elements when the step
      // of tree-reduction loop is not even.
      size_t PrevStep = WGSize;
      for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
        if (LID < CurStep)
          LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
        else if (!IsPow2WG && LID == CurStep && (PrevStep & 0x1))
          LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
        NDIt.barrier();
        PrevStep = CurStep;
      }

      // Compute the partial sum/reduction for the work-group.
      if (LID == 0) {
        size_t GrID = NDIt.get_group_linear_id();
        typename Reduction::result_type PSum =
            IsPow2WG ? LocalReds[0] : BOp(LocalReds[0], LocalReds[WGSize]);
        if (IsUpdateOfUserVar)
          PSum = BOp(*(Reduction::getOutPointer(Out)), PSum);
        Reduction::getOutPointer(Out)[GrID * NElements + E] = PSum;
      }

      // Ensure item 0 is finished with LocalReds before next iteration
      if (E != NElements - 1) {
        NDIt.barrier();
      }
    }
  });
}

namespace reduction {
namespace aux_krn {
template <class KernelName> struct FastReduce;
} // namespace aux_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: aux kernel, ext::oneapi::reduce(), reproducible results, FP +
/// ADD/MIN/MAX
template <typename KernelName, typename KernelType, class Reduction,
          typename InputT, typename OutputT>
void reduAuxCGFuncFastReduceImpl(handler &CGH, bool UniformWG,
                                 size_t NWorkItems, size_t NWorkGroups,
                                 size_t WGSize, Reduction &Redu, InputT In,
                                 OutputT Out) {
  constexpr size_t NElements = Reduction::num_elements;
  using Name =
      __sycl_reduction_kernel<reduction::aux_krn::FastReduce, KernelName>;
  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;
  range<1> GlobalRange = {UniformWG ? NWorkItems : NWorkGroups * WGSize};
  nd_range<1> Range{GlobalRange, range<1>(WGSize)};
  CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
    typename Reduction::binary_operation BOp;
    size_t WGID = NDIt.get_group_linear_id();
    size_t GID = NDIt.get_global_linear_id();

    for (int E = 0; E < NElements; ++E) {
      typename Reduction::result_type PSum =
          (UniformWG || (GID < NWorkItems))
              ? In[GID * NElements + E]
              : Reduction::reducer_type::getIdentity();
      PSum = reduce_over_group(NDIt.get_group(), PSum, BOp);
      if (NDIt.get_local_linear_id() == 0) {
        if (IsUpdateOfUserVar)
          PSum = BOp(Reduction::getOutPointer(Out)[E], PSum);
        Reduction::getOutPointer(Out)[WGID * NElements + E] = PSum;
      }
    }
  });
}

namespace reduction {
namespace aux_krn {
template <class KernelName> struct NoFastReduceNorAtomic;
} // namespace aux_krn
} // namespace reduction
/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups. At the end of each work-group the partial sum is written
/// to a global buffer.
///
/// Briefly: aux kernel, tree-reduction, CUSTOM types/ops.
template <typename KernelName, typename KernelType, class Reduction,
          typename InputT, typename OutputT>
void reduAuxCGFuncNoFastReduceNorAtomicImpl(handler &CGH, bool UniformPow2WG,
                                            size_t NWorkItems,
                                            size_t NWorkGroups, size_t WGSize,
                                            Reduction &Redu, InputT In,
                                            OutputT Out) {
  constexpr size_t NElements = Reduction::num_elements;
  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;

  // Use local memory to reduce elements in work-groups into 0-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch elements that could
  // otherwise be lost in the tree-reduction algorithm.
  size_t NumLocalElements = WGSize + (UniformPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NumLocalElements, CGH);

  auto ReduIdentity = Redu.getIdentity();
  auto BOp = Redu.getBinaryOperation();
  using Name =
      __sycl_reduction_kernel<reduction::aux_krn::NoFastReduceNorAtomic,
                              KernelName>;
  range<1> GlobalRange = {UniformPow2WG ? NWorkItems : NWorkGroups * WGSize};
  nd_range<1> Range{GlobalRange, range<1>(WGSize)};
  CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    size_t GID = NDIt.get_global_linear_id();

    for (int E = 0; E < NElements; ++E) {
      // Copy the element to local memory to prepare it for tree-reduction.
      LocalReds[LID] = (UniformPow2WG || GID < NWorkItems)
                           ? In[GID * NElements + E]
                           : ReduIdentity;
      if (!UniformPow2WG)
        LocalReds[WGSize] = ReduIdentity;
      NDIt.barrier();

      // Tree-reduction: reduce the local array LocalReds[:] to LocalReds[0]
      // LocalReds[WGSize] accumulates last/odd elements when the step
      // of tree-reduction loop is not even.
      size_t PrevStep = WGSize;
      for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
        if (LID < CurStep)
          LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
        else if (!UniformPow2WG && LID == CurStep && (PrevStep & 0x1))
          LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
        NDIt.barrier();
        PrevStep = CurStep;
      }

      // Compute the partial sum/reduction for the work-group.
      if (LID == 0) {
        size_t GrID = NDIt.get_group_linear_id();
        typename Reduction::result_type PSum =
            UniformPow2WG ? LocalReds[0] : BOp(LocalReds[0], LocalReds[WGSize]);
        if (IsUpdateOfUserVar)
          PSum = BOp(*(Reduction::getOutPointer(Out)), PSum);
        Reduction::getOutPointer(Out)[GrID * NElements + E] = PSum;
      }

      // Ensure item 0 is finished with LocalReds before next iteration
      if (E != NElements - 1) {
        NDIt.barrier();
      }
    }
  });
}

/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// At the end of each work-group the partial sum is written to a global buffer.
/// The function returns the number of the newly generated partial sums.
template <typename KernelName, typename KernelType, class Reduction>
size_t reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
                     Reduction &Redu) {
  constexpr size_t NElements = Reduction::num_elements;
  size_t NWorkGroups;
  size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

  // The last work-group may be not fully loaded with work, or the work group
  // size may be not power of two. Those two cases considered inefficient
  // as they require additional code and checks in the kernel.
  bool HasUniformWG = NWorkGroups * WGSize == NWorkItems;
  if (!Reduction::has_fast_reduce)
    HasUniformWG = HasUniformWG && (WGSize & (WGSize - 1)) == 0;

  // Get read accessor to the buffer that was used as output
  // in the previous kernel.
  auto In = Redu.getReadAccToPreviousPartialReds(CGH);
  auto Out = Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);

  if constexpr (Reduction::has_fast_reduce && !Reduction::has_fast_atomics) {
    reduAuxCGFuncFastReduceImpl<KernelName, KernelType>(
        CGH, HasUniformWG, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);

  } else {
    reduAuxCGFuncNoFastReduceNorAtomicImpl<KernelName, KernelType>(
        CGH, HasUniformWG, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);
  }
  return NWorkGroups;
}

// This method is used for implementation of parallel_for accepting 1 reduction.
// TODO: remove this method when everything is switched to general algorithm
// implementing arbitrary number of reductions in parallel_for().
/// Copies the final reduction result kept in read-write accessor to user's
/// accessor. This method is not called for user's read-write accessors
/// requiring update-write to it.
template <typename KernelName, class Reduction>
std::enable_if_t<!Reduction::is_usm>
reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu) {
  auto InAcc = Redu.getReadAccToPreviousPartialReds(CGH);
  associateWithHandler(CGH, &Redu.getUserRedVar(), access::target::device);
  CGH.copy(InAcc, Redu.getUserRedVar());
}

// This method is used for implementation of parallel_for accepting 1 reduction.
// TODO: remove this method when everything is switched to general algorithm
// implementing arbitrary number of reductions in parallel_for().
/// Copies the final reduction result kept in read-write accessor to user's
/// USM memory.
template <typename KernelName, class Reduction>
std::enable_if_t<Reduction::is_usm>
reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu) {
  constexpr size_t NElements = Reduction::num_elements;
  auto InAcc = Redu.getReadAccToPreviousPartialReds(CGH);
  auto UserVarPtr = Redu.getUserRedVar();
  bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
  auto BOp = Redu.getBinaryOperation();
  CGH.single_task<KernelName>([=] {
    for (int i = 0; i < NElements; ++i) {
      if (IsUpdateOfUserVar)
        UserVarPtr[i] = BOp(UserVarPtr[i], InAcc.get_pointer()[i]);
      else
        UserVarPtr[i] = InAcc.get_pointer()[i];
    }
  });
}

/// For the given 'Reductions' types pack and indices enumerating only
/// the reductions for which a local accessors are needed, this function creates
/// those local accessors and returns a tuple consisting of them.
template <typename... Reductions, size_t... Is>
auto createReduLocalAccs(size_t Size, handler &CGH,
                         std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::tuple_element_t<Is, std::tuple<Reductions...>>::getReadWriteLocalAcc(
          Size, CGH)...);
}

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

/// For the given 'Reductions' types pack and indices enumerating them this
/// function returns accessors to buffers holding partial sums generated in the
/// previous kernel invocation.
template <typename... Reductions, size_t... Is>
auto getReadAccsToPreviousPartialReds(handler &CGH,
                                      std::tuple<Reductions...> &ReduTuple,
                                      std::index_sequence<Is...>) {
  return makeReduTupleT(
      std::get<Is>(ReduTuple).getReadAccToPreviousPartialReds(CGH)...);
}

template <typename... Reductions, size_t... Is>
ReduTupleT<typename Reductions::result_type...>
getReduIdentities(std::tuple<Reductions...> &ReduTuple,
                  std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getIdentity()...};
}

template <typename... Reductions, size_t... Is>
ReduTupleT<typename Reductions::binary_operation...>
getReduBOPs(std::tuple<Reductions...> &ReduTuple, std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getBinaryOperation()...};
}

template <typename... Reductions, size_t... Is>
std::array<bool, sizeof...(Reductions)>
getInitToIdentityProperties(std::tuple<Reductions...> &ReduTuple,
                            std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).initializeToIdentity()...};
}

template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::reducer_type...>
createReducers(ReduTupleT<typename Reductions::result_type...> Identities,
               ReduTupleT<typename Reductions::binary_operation...> BOPsTuple,
               std::index_sequence<Is...>) {
  return {typename Reductions::reducer_type{std::get<Is>(Identities),
                                            std::get<Is>(BOPsTuple)}...};
}

template <typename KernelType, int Dims, typename... ReducerT, size_t... Is>
void callReduUserKernelFunc(KernelType KernelFunc, nd_item<Dims> NDIt,
                            std::tuple<ReducerT...> &Reducers,
                            std::index_sequence<Is...>) {
  KernelFunc(NDIt, std::get<Is>(Reducers)...);
}

template <typename... LocalAccT, typename... ReducerT, typename... ResultT,
          size_t... Is>
void initReduLocalAccs(bool Pow2WG, size_t LID, size_t WGSize,
                       ReduTupleT<LocalAccT...> LocalAccs,
                       const std::tuple<ReducerT...> &Reducers,
                       ReduTupleT<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[LID]...) =
      std::make_tuple(std::get<Is>(Reducers).MValue...);

  // For work-groups, which size is not power of two, local accessors have
  // an additional element with index WGSize that is used by the tree-reduction
  // algorithm. Initialize those additional elements with identity values here.
  if (!Pow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <typename... LocalAccT, typename... InputAccT, typename... ResultT,
          size_t... Is>
void initReduLocalAccs(bool UniformPow2WG, size_t LID, size_t GID,
                       size_t NWorkItems, size_t WGSize,
                       ReduTupleT<InputAccT...> LocalAccs,
                       ReduTupleT<LocalAccT...> InputAccs,
                       ReduTupleT<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  // Normally, the local accessors are initialized with elements from the input
  // accessors. The exception is the case when (GID >= NWorkItems), which
  // possible only when UniformPow2WG is false. For that case the elements of
  // local accessors are initialized with identity value, so they would not
  // give any impact into the final partial sums during the tree-reduction
  // algorithm work.
  if (UniformPow2WG || GID < NWorkItems)
    std::tie(std::get<Is>(LocalAccs)[LID]...) =
        std::make_tuple(std::get<Is>(InputAccs)[GID]...);
  else
    std::tie(std::get<Is>(LocalAccs)[LID]...) =
        std::make_tuple(std::get<Is>(Identities)...);

  // For work-groups, which size is not power of two, local accessors have
  // an additional element with index WGSize that is used by the tree-reduction
  // algorithm. Initialize those additional elements with identity values here.
  if (!UniformPow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <typename... LocalAccT, typename... BOPsT, size_t... Is>
void reduceReduLocalAccs(size_t IndexA, size_t IndexB,
                         ReduTupleT<LocalAccT...> LocalAccs,
                         ReduTupleT<BOPsT...> BOPs,
                         std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[IndexA]...) =
      std::make_tuple((std::get<Is>(BOPs)(std::get<Is>(LocalAccs)[IndexA],
                                          std::get<Is>(LocalAccs)[IndexB]))...);
}

template <typename... Reductions, typename... OutAccT, typename... LocalAccT,
          typename... BOPsT, typename... Ts, size_t... Is>
void writeReduSumsToOutAccs(
    bool Pow2WG, bool IsOneWG, size_t OutAccIndex, size_t WGSize,
    ReduTupleT<OutAccT...> OutAccs, ReduTupleT<LocalAccT...> LocalAccs,
    ReduTupleT<BOPsT...> BOPs, ReduTupleT<Ts...> IdentityVals,
    std::array<bool, sizeof...(Reductions)> IsInitializeToIdentity,
    std::index_sequence<Is...>) {
  // Add the initial value of user's variable to the final result.
  if (IsOneWG)
    std::tie(std::get<Is>(LocalAccs)[0]...) = std::make_tuple(std::get<Is>(
        BOPs)(std::get<Is>(LocalAccs)[0],
              IsInitializeToIdentity[Is]
                  ? std::get<Is>(IdentityVals)
                  : std::tuple_element_t<Is, std::tuple<Reductions...>>::
                        getOutPointer(std::get<Is>(OutAccs))[0])...);

  if (Pow2WG) {
    // The partial sums for the work-group are stored in 0-th elements of local
    // accessors. Simply write those sums to output accessors.
    std::tie(std::tuple_element_t<Is, std::tuple<Reductions...>>::getOutPointer(
        std::get<Is>(OutAccs))[OutAccIndex]...) =
        std::make_tuple(std::get<Is>(LocalAccs)[0]...);
  } else {
    // Each of local accessors keeps two partial sums: in 0-th and WGsize-th
    // elements. Combine them into final partial sums and write to output
    // accessors.
    std::tie(std::tuple_element_t<Is, std::tuple<Reductions...>>::getOutPointer(
        std::get<Is>(OutAccs))[OutAccIndex]...) =
        std::make_tuple(std::get<Is>(BOPs)(std::get<Is>(LocalAccs)[0],
                                           std::get<Is>(LocalAccs)[WGSize])...);
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

/// All scalar reductions are processed together; there is one loop of log2(N)
/// steps, and each reduction uses its own storage.
template <typename... Reductions, int Dims, typename... LocalAccT,
          typename... OutAccT, typename... ReducerT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduCGFuncImplScalar(
    bool Pow2WG, bool IsOneWG, nd_item<Dims> NDIt,
    ReduTupleT<LocalAccT...> LocalAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, std::tuple<ReducerT...> &ReducersTuple,
    ReduTupleT<Ts...> IdentitiesTuple, ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...> ReduIndices) {
  size_t WGSize = NDIt.get_local_range().size();
  size_t LID = NDIt.get_local_linear_id();
  initReduLocalAccs(Pow2WG, LID, WGSize, LocalAccsTuple, ReducersTuple,
                    IdentitiesTuple, ReduIndices);
  NDIt.barrier();

  size_t PrevStep = WGSize;
  for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
    if (LID < CurStep) {
      // LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
      reduceReduLocalAccs(LID, LID + CurStep, LocalAccsTuple, BOPsTuple,
                          ReduIndices);
    } else if (!Pow2WG && LID == CurStep && (PrevStep & 0x1)) {
      // LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
      reduceReduLocalAccs(WGSize, PrevStep - 1, LocalAccsTuple, BOPsTuple,
                          ReduIndices);
    }
    NDIt.barrier();
    PrevStep = CurStep;
  }

  // Compute the partial sum/reduction for the work-group.
  if (LID == 0) {
    size_t GrID = NDIt.get_group_linear_id();
    writeReduSumsToOutAccs<Reductions...>(
        Pow2WG, IsOneWG, GrID, WGSize, OutAccsTuple, LocalAccsTuple, BOPsTuple,
        IdentitiesTuple, InitToIdentityProps, ReduIndices);
  }
}

/// Each array reduction is processed separately.
template <typename Reduction, int Dims, typename LocalAccT, typename OutAccT,
          typename ReducerT, typename T, typename BOPT>
void reduCGFuncImplArrayHelper(bool Pow2WG, bool IsOneWG, nd_item<Dims> NDIt,
                               LocalAccT LocalReds, OutAccT Out,
                               ReducerT &Reducer, T Identity, BOPT BOp,
                               bool IsInitializeToIdentity) {
  size_t WGSize = NDIt.get_local_range().size();
  size_t LID = NDIt.get_local_linear_id();

  // If there are multiple values, reduce each separately
  // This prevents local memory from scaling with elements
  auto NElements = Reduction::num_elements;
  for (size_t E = 0; E < NElements; ++E) {

    // Copy the element to local memory to prepare it for tree-reduction.
    LocalReds[LID] = Reducer.getElement(E);
    if (!Pow2WG)
      LocalReds[WGSize] = Identity;
    NDIt.barrier();

    size_t PrevStep = WGSize;
    for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
      if (LID < CurStep) {
        LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
      } else if (!Pow2WG && LID == CurStep && (PrevStep & 0x1)) {
        LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
      }
      NDIt.barrier();
      PrevStep = CurStep;
    }

    // Add the initial value of user's variable to the final result.
    if (LID == 0) {
      if (IsOneWG) {
        LocalReds[0] =
            BOp(LocalReds[0], IsInitializeToIdentity
                                  ? Identity
                                  : Reduction::getOutPointer(Out)[E]);
      }

      size_t GrID = NDIt.get_group_linear_id();
      if (Pow2WG) {
        // The partial sums for the work-group are stored in 0-th elements of
        // local accessors. Simply write those sums to output accessors.
        Reduction::getOutPointer(Out)[GrID * NElements + E] = LocalReds[0];
      } else {
        // Each of local accessors keeps two partial sums: in 0-th and WGsize-th
        // elements. Combine them into final partial sums and write to output
        // accessors.
        Reduction::getOutPointer(Out)[GrID * NElements + E] =
            BOp(LocalReds[0], LocalReds[WGSize]);
      }
    }

    // Ensure item 0 is finished with LocalReds before next iteration
    if (E != NElements - 1) {
      NDIt.barrier();
    }
  }
}

template <typename... Reductions, int Dims, typename... LocalAccT,
          typename... OutAccT, typename... ReducerT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduCGFuncImplArray(
    bool Pow2WG, bool IsOneWG, nd_item<Dims> NDIt,
    ReduTupleT<LocalAccT...> LocalAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, std::tuple<ReducerT...> &ReducersTuple,
    ReduTupleT<Ts...> IdentitiesTuple, ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...>) {
  using ReductionPack = std::tuple<Reductions...>;
  (reduCGFuncImplArrayHelper<std::tuple_element_t<Is, ReductionPack>>(
       Pow2WG, IsOneWG, NDIt, std::get<Is>(LocalAccsTuple),
       std::get<Is>(OutAccsTuple), std::get<Is>(ReducersTuple),
       std::get<Is>(IdentitiesTuple), std::get<Is>(BOPsTuple),
       InitToIdentityProps[Is]),
   ...);
}

namespace reduction {
namespace main_krn {
template <class KernelName, class Accessor> struct NDRangeMulti;
} // namespace main_krn
} // namespace reduction
template <typename KernelName, typename KernelType, int Dims,
          typename... Reductions, size_t... Is>
void reduCGFuncMulti(handler &CGH, KernelType KernelFunc,
                     const nd_range<Dims> &Range,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...> ReduIndices) {
  size_t WGSize = Range.get_local_range().size();
  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;

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

  // Create inputs using the global order of all reductions
  size_t LocalAccSize = WGSize + (Pow2WG ? 0 : 1);
  auto LocalAccsTuple =
      createReduLocalAccs<Reductions...>(LocalAccSize, CGH, ReduIndices);

  size_t NWorkGroups = Range.get_group_range().size();
  bool IsOneWG = NWorkGroups == 1;

  // The type of the Out "accessor" differs between scenarios when there is just
  // one WorkGroup and when there are multiple. Use this lambda to write the
  // code just once.
  auto Rest = [&](auto OutAccsTuple) {
    auto IdentitiesTuple = getReduIdentities(ReduTuple, ReduIndices);
    auto BOPsTuple = getReduBOPs(ReduTuple, ReduIndices);
    auto InitToIdentityProps =
        getInitToIdentityProperties(ReduTuple, ReduIndices);

    using Name = __sycl_reduction_kernel<reduction::main_krn::NDRangeMulti,
                                         KernelName, decltype(OutAccsTuple)>;
    CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
      // Pass all reductions to user's lambda in the same order as supplied
      // Each reducer initializes its own storage
      auto ReduIndices = std::index_sequence_for<Reductions...>();
      auto ReducersTuple = createReducers<Reductions...>(
          IdentitiesTuple, BOPsTuple, ReduIndices);
      callReduUserKernelFunc(KernelFunc, NDIt, ReducersTuple, ReduIndices);

      // Combine and write-back the results of any scalar reductions
      // reduCGFuncImplScalar<Reductions...>(NDIt, LocalAccsTuple, OutAccsTuple,
      // ReducersTuple, IdentitiesTuple, BOPsTuple, InitToIdentityProps,
      // ReduIndices);
      reduCGFuncImplScalar<Reductions...>(
          Pow2WG, IsOneWG, NDIt, LocalAccsTuple, OutAccsTuple, ReducersTuple,
          IdentitiesTuple, BOPsTuple, InitToIdentityProps, ScalarIs);

      // Combine and write-back the results of any array reductions
      // These are handled separately to minimize temporary storage and account
      // for the fact that each array reduction may have a different number of
      // elements to reduce (i.e. a different extent).
      reduCGFuncImplArray<Reductions...>(
          Pow2WG, IsOneWG, NDIt, LocalAccsTuple, OutAccsTuple, ReducersTuple,
          IdentitiesTuple, BOPsTuple, InitToIdentityProps, ArrayIs);
    });
  };

  if (IsOneWG)
    Rest(createReduOutAccs<true>(NWorkGroups, CGH, ReduTuple, ReduIndices));
  else
    Rest(createReduOutAccs<false>(NWorkGroups, CGH, ReduTuple, ReduIndices));
}

namespace reduction {
namespace main_krn {
template <class KernelName> struct NDRangeAtomic64;
} // namespace main_krn
} // namespace reduction

// Specialization for devices with the atomic64 aspect, which guarantees 64 bit
// floating point support for atomic reduction operation.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFuncAtomic64(handler &CGH, KernelType KernelFunc,
                        const nd_range<Dims> &Range, Reduction &Redu) {
  auto Out = Redu.getReadWriteAccessorToInitializedMem(CGH);
  static_assert(
      Reduction::has_float64_atomics,
      "Only suitable for reductions that have FP64 atomic operations.");
  constexpr size_t NElements = Reduction::num_elements;
  using Name =
      __sycl_reduction_kernel<reduction::main_krn::NDRangeAtomic64, KernelName>;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's function. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    // If there are multiple values, reduce each separately
    // reduce_over_group is only defined for each T, not for span<T, ...>
    for (int E = 0; E < NElements; ++E) {
      typename Reduction::binary_operation BOp;
      Reducer.getElement(E) =
          reduce_over_group(NDIt.get_group(), Reducer.getElement(E), BOp);
    }

    if (NDIt.get_local_linear_id() == 0) {
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
    }
  });
}

template <typename... Reductions, size_t... Is>
void associateReduAccsWithHandler(handler &CGH,
                                  std::tuple<Reductions...> &ReduTuple,
                                  std::index_sequence<Is...>) {
  auto ProcessOne = [&CGH](auto Redu) {
    if constexpr (decltype(Redu)::is_acc) {
      associateWithHandler(CGH, &Redu.getUserRedVar(), access::target::device);
    }
  };
  (ProcessOne(std::get<Is>(ReduTuple)), ...);
}

/// All scalar reductions are processed together; there is one loop of log2(N)
/// steps, and each reduction uses its own storage.
template <typename... Reductions, int Dims, typename... LocalAccT,
          typename... InAccT, typename... OutAccT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduAuxCGFuncImplScalar(
    bool UniformPow2WG, bool IsOneWG, nd_item<Dims> NDIt, size_t LID,
    size_t GID, size_t NWorkItems, size_t WGSize,
    ReduTupleT<LocalAccT...> LocalAccsTuple, ReduTupleT<InAccT...> InAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, ReduTupleT<Ts...> IdentitiesTuple,
    ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...> ReduIndices) {
  initReduLocalAccs(UniformPow2WG, LID, GID, NWorkItems, WGSize, LocalAccsTuple,
                    InAccsTuple, IdentitiesTuple, ReduIndices);
  NDIt.barrier();

  size_t PrevStep = WGSize;
  for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
    if (LID < CurStep) {
      // LocalAcc[LID] = BOp(LocalAcc[LID], LocalAcc[LID + CurStep]);
      reduceReduLocalAccs(LID, LID + CurStep, LocalAccsTuple, BOPsTuple,
                          ReduIndices);
    } else if (!UniformPow2WG && LID == CurStep && (PrevStep & 0x1)) {
      // LocalAcc[WGSize] = BOp(LocalAcc[WGSize], LocalAcc[PrevStep - 1]);
      reduceReduLocalAccs(WGSize, PrevStep - 1, LocalAccsTuple, BOPsTuple,
                          ReduIndices);
    }
    NDIt.barrier();
    PrevStep = CurStep;
  }

  // Compute the partial sum/reduction for the work-group.
  if (LID == 0) {
    size_t GrID = NDIt.get_group_linear_id();
    writeReduSumsToOutAccs<Reductions...>(
        UniformPow2WG, IsOneWG, GrID, WGSize, OutAccsTuple, LocalAccsTuple,
        BOPsTuple, IdentitiesTuple, InitToIdentityProps, ReduIndices);
  }
}

template <typename Reduction, int Dims, typename LocalAccT, typename InAccT,
          typename OutAccT, typename T, typename BOPT>
void reduAuxCGFuncImplArrayHelper(bool UniformPow2WG, bool IsOneWG,
                                  nd_item<Dims> NDIt, size_t LID, size_t GID,
                                  size_t NWorkItems, size_t WGSize,
                                  LocalAccT LocalReds, InAccT In, OutAccT Out,
                                  T Identity, BOPT BOp,
                                  bool IsInitializeToIdentity) {

  // If there are multiple values, reduce each separately
  // This prevents local memory from scaling with elements
  auto NElements = Reduction::num_elements;
  for (size_t E = 0; E < NElements; ++E) {
    // Normally, the local accessors are initialized with elements from the
    // input accessors. The exception is the case when (GID >= NWorkItems),
    // which possible only when UniformPow2WG is false. For that case the
    // elements of local accessors are initialized with identity value, so they
    // would not give any impact into the final partial sums during the
    // tree-reduction algorithm work.
    if (UniformPow2WG || GID < NWorkItems) {
      LocalReds[LID] = In[GID * NElements + E];
    } else {
      LocalReds[LID] = Identity;
    }

    // For work-groups, which size is not power of two, local accessors have
    // an additional element with index WGSize that is used by the
    // tree-reduction algorithm. Initialize those additional elements with
    // identity values here.
    if (!UniformPow2WG) {
      LocalReds[WGSize] = Identity;
    }

    NDIt.barrier();

    // Tree reduction in local memory
    size_t PrevStep = WGSize;
    for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
      if (LID < CurStep) {
        LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
      } else if (!UniformPow2WG && LID == CurStep && (PrevStep & 0x1)) {
        LocalReds[WGSize] = BOp(LocalReds[WGSize], LocalReds[PrevStep - 1]);
      }
      NDIt.barrier();
      PrevStep = CurStep;
    }

    // Add the initial value of user's variable to the final result.
    if (LID == 0) {
      if (IsOneWG) {
        LocalReds[0] =
            BOp(LocalReds[0], IsInitializeToIdentity
                                  ? Identity
                                  : Reduction::getOutPointer(Out)[E]);
      }

      size_t GrID = NDIt.get_group_linear_id();
      if (UniformPow2WG) {
        // The partial sums for the work-group are stored in 0-th elements of
        // local accessors. Simply write those sums to output accessors.
        Reduction::getOutPointer(Out)[GrID * NElements + E] = LocalReds[0];
      } else {
        // Each of local accessors keeps two partial sums: in 0-th and WGsize-th
        // elements. Combine them into final partial sums and write to output
        // accessors.
        Reduction::getOutPointer(Out)[GrID * NElements + E] =
            BOp(LocalReds[0], LocalReds[WGSize]);
      }
    }

    // Ensure item 0 is finished with LocalReds before next iteration
    if (E != NElements - 1) {
      NDIt.barrier();
    }
  }
}

template <typename... Reductions, int Dims, typename... LocalAccT,
          typename... InAccT, typename... OutAccT, typename... Ts,
          typename... BOPsT, size_t... Is>
void reduAuxCGFuncImplArray(
    bool UniformPow2WG, bool IsOneWG, nd_item<Dims> NDIt, size_t LID,
    size_t GID, size_t NWorkItems, size_t WGSize,
    ReduTupleT<LocalAccT...> LocalAccsTuple, ReduTupleT<InAccT...> InAccsTuple,
    ReduTupleT<OutAccT...> OutAccsTuple, ReduTupleT<Ts...> IdentitiesTuple,
    ReduTupleT<BOPsT...> BOPsTuple,
    std::array<bool, sizeof...(Reductions)> InitToIdentityProps,
    std::index_sequence<Is...>) {
  using ReductionPack = std::tuple<Reductions...>;
  (reduAuxCGFuncImplArrayHelper<std::tuple_element_t<Is, ReductionPack>>(
       UniformPow2WG, IsOneWG, NDIt, LID, GID, NWorkItems, WGSize,
       std::get<Is>(LocalAccsTuple), std::get<Is>(InAccsTuple),
       std::get<Is>(OutAccsTuple), std::get<Is>(IdentitiesTuple),
       std::get<Is>(BOPsTuple), InitToIdentityProps[Is]),
   ...);
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFunc(handler &CGH, KernelType KernelFunc,
                const nd_range<Dims> &Range, Reduction &Redu) {
  size_t WGSize = Range.get_local_range().size();
  auto Out = [&]() {
    if constexpr (Reduction::has_fast_atomics) {

      // User's initialized read-write accessor is re-used here if
      // initialize_to_identity is not set (i.e. if user's variable is
      // initialized). Otherwise, a new buffer is initialized with identity
      // value and a new read-write accessor to that buffer is created. That is
      // done because atomic operations update some initialized memory. User's
      // USM pointer is not re-used even when initialize_to_identity is not set
      // because it does not worth the creation of an additional variant of a
      // user's kernel for that case.
      return Redu.getReadWriteAccessorToInitializedMem(CGH);

    } else {
      constexpr size_t NElements = Reduction::num_elements;
      size_t NWorkGroups = Range.get_group_range().size();

      return Redu.getWriteAccForPartialReds(NWorkGroups * NElements, CGH);
    }
  }();

  if constexpr (Reduction::has_fast_reduce) {
    if constexpr (Reduction::has_fast_atomics) {
      reduCGFuncForNDRangeBothFastReduceAndAtomics<KernelName, KernelType>(
          CGH, KernelFunc, Range, Redu, Out);
    } else {
      reduCGFuncForNDRangeFastReduceOnly<KernelName, KernelType>(
          CGH, KernelFunc, Range, Redu, Out);
    }
  } else {
    bool IsPow2WG = (WGSize & (WGSize - 1)) == 0;
    if constexpr (Reduction::has_fast_atomics) {
      reduCGFuncForNDRangeFastAtomicsOnly<KernelName, KernelType>(
          CGH, IsPow2WG, KernelFunc, Range, Redu, Out);
    } else {
      reduCGFuncForNDRangeBasic<KernelName, KernelType>(
          CGH, IsPow2WG, KernelFunc, Range, Redu, Out);
    }
  }
}

namespace reduction {
namespace aux_krn {
template <class KernelName, class Accessor> struct Multi;
} // namespace aux_krn
} // namespace reduction
template <typename KernelName, typename KernelType, typename... Reductions,
          size_t... Is>
size_t reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...> ReduIndices) {
  size_t NWorkGroups;
  size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;
  bool IsOneWG = NWorkGroups == 1;
  bool HasUniformWG = Pow2WG && (NWorkGroups * WGSize == NWorkItems);

  // Like reduCGFuncImpl, we also have to split out scalar and array reductions
  IsScalarReduction ScalarPredicate;
  auto ScalarIs = filterSequence<Reductions...>(ScalarPredicate, ReduIndices);

  IsArrayReduction ArrayPredicate;
  auto ArrayIs = filterSequence<Reductions...>(ArrayPredicate, ReduIndices);

  size_t LocalAccSize = WGSize + (HasUniformWG ? 0 : 1);
  auto LocalAccsTuple =
      createReduLocalAccs<Reductions...>(LocalAccSize, CGH, ReduIndices);
  auto InAccsTuple =
      getReadAccsToPreviousPartialReds(CGH, ReduTuple, ReduIndices);

  auto IdentitiesTuple = getReduIdentities(ReduTuple, ReduIndices);
  auto BOPsTuple = getReduBOPs(ReduTuple, ReduIndices);
  auto InitToIdentityProps =
      getInitToIdentityProperties(ReduTuple, ReduIndices);

  // Predicate/OutAccsTuple below have different type depending on us having
  // just a single WG or multiple WGs. Use this lambda to avoid code
  // duplication.
  auto Rest = [&](auto Predicate, auto OutAccsTuple) {
    auto AccReduIndices = filterSequence<Reductions...>(Predicate, ReduIndices);
    associateReduAccsWithHandler(CGH, ReduTuple, AccReduIndices);
    using Name = __sycl_reduction_kernel<reduction::aux_krn::Multi, KernelName,
                                         decltype(OutAccsTuple)>;
    // TODO: Opportunity to parallelize across number of elements
    range<1> GlobalRange = {HasUniformWG ? NWorkItems : NWorkGroups * WGSize};
    nd_range<1> Range{GlobalRange, range<1>(WGSize)};
    CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
      size_t WGSize = NDIt.get_local_range().size();
      size_t LID = NDIt.get_local_linear_id();
      size_t GID = NDIt.get_global_linear_id();

      // Handle scalar and array reductions
      reduAuxCGFuncImplScalar<Reductions...>(
          HasUniformWG, IsOneWG, NDIt, LID, GID, NWorkItems, WGSize,
          LocalAccsTuple, InAccsTuple, OutAccsTuple, IdentitiesTuple, BOPsTuple,
          InitToIdentityProps, ScalarIs);
      reduAuxCGFuncImplArray<Reductions...>(
          HasUniformWG, IsOneWG, NDIt, LID, GID, NWorkItems, WGSize,
          LocalAccsTuple, InAccsTuple, OutAccsTuple, IdentitiesTuple, BOPsTuple,
          InitToIdentityProps, ArrayIs);
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

inline void
reduSaveFinalResultToUserMemHelper(std::vector<event> &,
                                   std::shared_ptr<detail::queue_impl>, bool) {}

template <typename Reduction, typename... RestT>
void reduSaveFinalResultToUserMemHelper(
    std::vector<event> &Events, std::shared_ptr<detail::queue_impl> Queue,
    bool IsHost, Reduction &Redu, RestT... Rest) {
  if constexpr (Reduction::is_dw_acc) {
    event CopyEvent = withAuxHandler(Queue, IsHost, [&](handler &CopyHandler) {
      auto InAcc = Redu.getReadAccToPreviousPartialReds(CopyHandler);
      auto OutAcc = Redu.getUserRedVar();
      associateWithHandler(CopyHandler, &Redu.getUserRedVar(),
                           access::target::device);
      if (!Events.empty())
        CopyHandler.depends_on(Events.back());
      CopyHandler.copy(InAcc, OutAcc);
    });
    Events.push_back(CopyEvent);
  }
  reduSaveFinalResultToUserMemHelper(Events, Queue, IsHost, Rest...);
}

/// Creates additional kernels that copy the accumulated/final results from
/// reductions accessors to either user's accessor or user's USM memory.
/// Returns the event to the last kernel copying data or nullptr if no
/// additional kernels created.
template <typename... Reduction, size_t... Is>
std::shared_ptr<event>
reduSaveFinalResultToUserMem(std::shared_ptr<detail::queue_impl> Queue,
                             bool IsHost, std::tuple<Reduction...> &ReduTuple,
                             std::index_sequence<Is...>) {
  std::vector<event> Events;
  reduSaveFinalResultToUserMemHelper(Events, Queue, IsHost,
                                     std::get<Is>(ReduTuple)...);
  if (!Events.empty())
    return std::make_shared<event>(Events.back());
  return std::shared_ptr<event>();
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

} // namespace detail

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 3 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc, identity value \param Identity, and the binary
/// operation used in the reduction.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH>
auto reduction(accessor<T, Dims, AccMode, access::target::device, IsPH> &Acc,
               const T &Identity, BinaryOperation BOp) {
  return detail::make_reduction<BinaryOperation, 0, 1>(Acc, Identity, BOp);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc and the binary operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH,
          typename = std::enable_if_t<
              sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
auto reduction(accessor<T, Dims, AccMode, access::target::device, IsPH> &Acc,
               BinaryOperation) {
  return detail::make_reduction<BinaryOperation, 0, 1>(Acc);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 3 arguments: the reference to the reduction variable to where
/// the computed reduction must be stored \param VarPtr, identity value
/// \param Identity, and the binary operation used in the reduction.
template <typename T, class BinaryOperation>
auto reduction(T *VarPtr, const T &Identity, BinaryOperation BOp) {
  return detail::make_reduction<BinaryOperation, 0, 1>(VarPtr, Identity, BOp);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the reference to the reduction variable, to where
/// the computed reduction must be stored \param VarPtr, and the binary
/// operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::detail::IsKnownIdentityOp<T, BinaryOperation>::value>>
auto reduction(T *VarPtr, BinaryOperation) {
  return detail::make_reduction<BinaryOperation, 0, 1>(VarPtr);
}

// ---- has_known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity
    : sycl::has_known_identity<BinaryOperation, AccumulatorT> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

// ---- known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity : sycl::known_identity<BinaryOperation, AccumulatorT> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace oneapi
} // namespace ext

#ifdef __SYCL_INTERNAL_API
namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
using namespace ext::oneapi;
namespace detail {
using sycl::detail::queue_impl;
__SYCL_EXPORT size_t reduGetMaxWGSize(std::shared_ptr<queue_impl> Queue,
                                      size_t LocalMemBytesPerWorkItem);
__SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                       size_t &NWorkGroups);
} // namespace detail
} // namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead")ONEAPI
#endif // __SYCL_INTERNAL_API
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif // __cplusplus >= 201703L
