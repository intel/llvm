//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/detail/tuple.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/known_identity.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/ext/oneapi/group_algorithm.hpp>

#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

namespace detail {

using cl::sycl::detail::bool_constant;
using cl::sycl::detail::enable_if_t;
using cl::sycl::detail::queue_impl;
using cl::sycl::detail::remove_AS;

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
    bool_constant<sycl::detail::is_sgeninteger<T>::value &&
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
// TODO 32 bit floating point atomics are eventually expected to be supported by
// the has_fast_atomics specialization. Once the reducer class is updated to
// replace the deprecated atomic class with atomic_ref, the (sizeof(T) == 4)
// case should be removed here and replaced in IsReduOptForFastAtomicFetch.
template <typename T, class BinaryOperation>
using IsReduOptForAtomic64Add =
#ifdef SYCL_REDUCTION_DETERMINISTIC
    bool_constant<false>;
#else
    bool_constant<sycl::detail::IsPlus<T, BinaryOperation>::value &&
                  sycl::detail::is_sgenfloat<T>::value &&
                  (sizeof(T) == 4 || sizeof(T) == 8)>;
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
template <typename T, class BinaryOperation, typename Subst = void>
class reducer {
public:
  reducer(const T &Identity, BinaryOperation BOp)
      : MValue(Identity), MIdentity(Identity), MBinaryOp(BOp) {}
  void combine(const T &Partial) { MValue = MBinaryOp(MValue, Partial); }

  T getIdentity() const { return MIdentity; }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++() {
    combine(static_cast<T>(1));
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++(int) {
    combine(static_cast<T>(1));
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value>
  operator+=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsMultiplies<_T, BinaryOperation>::value>
  operator*=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitOR<_T, BinaryOperation>::value>
  operator|=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitXOR<_T, BinaryOperation>::value>
  operator^=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitAND<_T, BinaryOperation>::value>
  operator&=(const _T &Partial) {
    combine(Partial);
  }

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
//
// TODO: More types and ops can be added to here later.
template <typename T, class BinaryOperation>
class reducer<T, BinaryOperation,
              enable_if_t<IsKnownIdentityOp<T, BinaryOperation>::value>> {
public:
  reducer() : MValue(getIdentity()) {}
  reducer(const T &, BinaryOperation) : MValue(getIdentity()) {}

  void combine(const T &Partial) {
    BinaryOperation BOp;
    MValue = BOp(MValue, Partial);
  }

  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<has_known_identity_impl<_BinaryOperation, _T>::value, _T>
  getIdentity() {
    return known_identity_impl<_BinaryOperation, _T>::value;
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++() {
    combine(static_cast<T>(1));
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value &&
              sycl::detail::is_geninteger<_T>::value>
  operator++(int) {
    combine(static_cast<T>(1));
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsPlus<_T, BinaryOperation>::value>
  operator+=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsMultiplies<_T, BinaryOperation>::value>
  operator*=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitOR<_T, BinaryOperation>::value>
  operator|=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitXOR<_T, BinaryOperation>::value>
  operator^=(const _T &Partial) {
    combine(Partial);
  }

  template <typename _T = T>
  enable_if_t<sycl::detail::IsBitAND<_T, BinaryOperation>::value>
  operator&=(const _T &Partial) {
    combine(Partial);
  }

private:
  template <access::address_space Space>
  static constexpr memory_scope getMemoryScope() {
    return Space == access::address_space::local_space
               ? memory_scope::work_group
               : memory_scope::device;
  }

public:
  /// Atomic ADD operation: *ReduVarPtr += MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value ||
               IsReduOptForAtomic64Add<T, _BinaryOperation>::value) &&
              sycl::detail::IsPlus<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_add(MValue);
  }

  /// Atomic BITWISE OR operation: *ReduVarPtr |= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsBitOR<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_or(MValue);
  }

  /// Atomic BITWISE XOR operation: *ReduVarPtr ^= MValue;
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsBitXOR<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_xor(MValue);
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
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_and(MValue);
  }

  /// Atomic MIN operation: *ReduVarPtr = sycl::minimum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsMinimum<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_min(MValue);
  }

  /// Atomic MAX operation: *ReduVarPtr = sycl::maximum(*ReduVarPtr, MValue);
  template <access::address_space Space = access::address_space::global_space,
            typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              IsReduOptForFastAtomicFetch<T, _BinaryOperation>::value &&
              sycl::detail::IsMaximum<T, _BinaryOperation>::value &&
              (Space == access::address_space::global_space ||
               Space == access::address_space::local_space)>
  atomic_combine(_T *ReduVarPtr) const {
    atomic_ref<T, memory_order::relaxed, getMemoryScope<Space>(), Space>(
        *multi_ptr<T, Space>(ReduVarPtr))
        .fetch_max(MValue);
  }

  T MValue;
};

/// Base non-template class which is a base class for all reduction
/// implementation classes. It is needed to detect the reduction classes.
class reduction_impl_base {};

/// Predicate returning true if all template type parameters except the last one
/// are reductions.
template <typename FirstT, typename... RestT> struct AreAllButLastReductions {
  static constexpr bool value =
      std::is_base_of<reduction_impl_base, FirstT>::value &&
      AreAllButLastReductions<RestT...>::value;
};

/// Helper specialization of AreAllButLastReductions for one element only.
/// Returns true if the template parameter is not a reduction.
template <typename T> struct AreAllButLastReductions<T> {
  static constexpr bool value = !std::is_base_of<reduction_impl_base, T>::value;
};

/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, bool IsUSM,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class reduction_impl : private reduction_impl_base {
public:
  using reducer_type = reducer<T, BinaryOperation>;
  using result_type = T;
  using binary_operation = BinaryOperation;
  using rw_accessor_type =
      accessor<T, Dims, access::mode::read_write, access::target::global_buffer,
               IsPlaceholder, ext::oneapi::accessor_property_list<>>;
  using dw_accessor_type =
      accessor<T, Dims, access::mode::discard_write,
               access::target::global_buffer, IsPlaceholder,
               ext::oneapi::accessor_property_list<>>;
  static constexpr int accessor_dim = Dims;
  static constexpr int buffer_dim = (Dims == 0) ? 1 : Dims;

  static constexpr bool has_atomic_add_float64 =
      IsReduOptForAtomic64Add<T, BinaryOperation>::value;
  static constexpr bool has_fast_atomics =
      IsReduOptForFastAtomicFetch<T, BinaryOperation>::value;
  static constexpr bool has_fast_reduce =
      IsReduOptForFastReduce<T, BinaryOperation>::value;
  static constexpr bool is_usm = IsUSM;
  static constexpr bool is_placeholder =
      (IsPlaceholder == access::placeholder::true_t);

  // Only scalar (i.e. 0-dim and 1-dim with 1 element) reductions supported now.
  // TODO: suport (Dims > 1) accessors/reductions.
  // TODO: support true 1-Dimensional accessors/reductions (size() > 1).
  // (size() == 1) is checked in the constructor of reduction_impl.
  static_assert(Dims <= 1,
                "Multi-dimensional reductions are not supported yet.");

  /// Returns the statically known identity value.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value,
              _T> constexpr getIdentity() {
    return reducer_type::getIdentity();
  }

  /// Returns the identity value given by user.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<!IsKnownIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return MIdentity;
  }

  /// SYCL-2020.
  /// Constructs reduction_impl when the identity value is statically known.
  template <typename _T, typename AllocatorT,
            std::enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * =
                nullptr>
  reduction_impl(buffer<_T, 1, AllocatorT> Buffer, handler &CGH,
                 bool InitializeToIdentity)
      : MRWAcc(std::make_shared<rw_accessor_type>(Buffer)),
        MIdentity(getIdentity()), InitializeToIdentity(InitializeToIdentity) {
    associateWithHandler(CGH);
    if (Buffer.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is statically known.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(rw_accessor_type &Acc)
      : MRWAcc(new rw_accessor_type(Acc)), MIdentity(getIdentity()),
        InitializeToIdentity(false) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is statically known.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(dw_accessor_type &Acc)
      : MDWAcc(new dw_accessor_type(Acc)), MIdentity(getIdentity()),
        InitializeToIdentity(true) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// SYCL-2020.
  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  template <
      typename _T, typename AllocatorT,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(buffer<_T, 1, AllocatorT> Buffer, handler &CGH,
                 const T & /*Identity*/, BinaryOperation,
                 bool InitializeToIdentity)
      : MRWAcc(std::make_shared<rw_accessor_type>(Buffer)),
        MIdentity(getIdentity()), InitializeToIdentity(InitializeToIdentity) {
    associateWithHandler(CGH);
    if (Buffer.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
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
  }

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(rw_accessor_type &Acc, const T & /*Identity*/, BinaryOperation)
      : MRWAcc(new rw_accessor_type(Acc)), MIdentity(getIdentity()),
        InitializeToIdentity(false) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
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
  }

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(dw_accessor_type &Acc, const T & /*Identity*/, BinaryOperation)
      : MDWAcc(new dw_accessor_type(Acc)), MIdentity(getIdentity()),
        InitializeToIdentity(true) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
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
  }

  /// SYCL-2020.
  /// Constructs reduction_impl when the identity value is NOT known statically.
  template <
      typename _T, typename AllocatorT,
      enable_if_t<!IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(buffer<_T, 1, AllocatorT> Buffer, handler &CGH,
                 const T &Identity, BinaryOperation BOp,
                 bool InitializeToIdentity)
      : MRWAcc(std::make_shared<rw_accessor_type>(Buffer)), MIdentity(Identity),
        MBinaryOp(BOp), InitializeToIdentity(InitializeToIdentity) {
    associateWithHandler(CGH);
    if (Buffer.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is unknown.
  template <
      typename _T = T,
      enable_if_t<!IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(rw_accessor_type &Acc, const T &Identity, BinaryOperation BOp)
      : MRWAcc(new rw_accessor_type(Acc)), MIdentity(Identity), MBinaryOp(BOp),
        InitializeToIdentity(false) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is unknown.
  template <
      typename _T = T,
      enable_if_t<!IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(dw_accessor_type &Acc, const T &Identity, BinaryOperation BOp)
      : MDWAcc(new dw_accessor_type(Acc)), MIdentity(Identity), MBinaryOp(BOp),
        InitializeToIdentity(true) {
    if (Acc.size() != 1)
      throw sycl::runtime_error("Reduction variable must be a scalar.",
                                PI_INVALID_VALUE);
  }

  /// Constructs reduction_impl when the identity value is statically known.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr, bool InitializeToIdentity = false)
      : MIdentity(getIdentity()), MUSMPointer(VarPtr),
        InitializeToIdentity(InitializeToIdentity) {}

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <
      typename _T = T,
      enable_if_t<IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr, const T &Identity, BinaryOperation,
                 bool InitializeToIdentity = false)
      : MIdentity(Identity), MUSMPointer(VarPtr),
        InitializeToIdentity(InitializeToIdentity) {
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
  }

  /// Constructs reduction_impl when the identity value is unknown.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <
      typename _T = T,
      enable_if_t<!IsKnownIdentityOp<_T, BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr, const T &Identity, BinaryOperation BOp,
                 bool InitializeToIdentity = false)
      : MIdentity(Identity), MUSMPointer(VarPtr), MBinaryOp(BOp),
        InitializeToIdentity(InitializeToIdentity) {}

  /// Associates the reduction accessor to user's memory with \p CGH handler
  /// to keep the accessor alive until the command group finishes the work.
  /// This function does not do anything for USM reductions.
  void associateWithHandler(handler &CGH) {
#ifndef __SYCL_DEVICE_ONLY__
    if (MRWAcc)
      CGH.associateWithHandler(MRWAcc.get(), access::target::global_buffer);
    else if (MDWAcc)
      CGH.associateWithHandler(MDWAcc.get(), access::target::global_buffer);
#else
    (void)CGH;
#endif
  }

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

  /// Returns user's USM pointer passed to reduction for editing.
  template <bool IsOneWG, bool _IsUSM = is_usm>
  std::enable_if_t<IsOneWG && _IsUSM, result_type *>
  getWriteMemForPartialReds(size_t, handler &) {
    return getUSMPointer();
  }

  /// Returns user's accessor passed to reduction for editing if that is
  /// the read-write accessor. Otherwise, create a new buffer and return
  /// read-write accessor to it.
  template <bool IsOneWG, bool _IsUSM = is_usm>
  std::enable_if_t<IsOneWG && !_IsUSM, rw_accessor_type>
  getWriteMemForPartialReds(size_t, handler &CGH) {
    if (MRWAcc)
      return *MRWAcc;
    return getWriteMemForPartialReds<false>(1, CGH);
  }

  /// Constructs a new temporary buffer to hold partial sums and returns
  /// the accessor for that buffer.
  template <bool IsOneWG>
  std::enable_if_t<!IsOneWG, rw_accessor_type>
  getWriteMemForPartialReds(size_t Size, handler &CGH) {
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
  }

  /// Returns an accessor accessing the memory that will hold the reduction
  /// partial sums.
  /// If \p Size is equal to one, then the reduction result is the final and
  /// needs to be written to user's read-write accessor (if there is such).
  /// Otherwise, a new buffer is created and accessor to that buffer is
  /// returned.
  rw_accessor_type getWriteAccForPartialReds(size_t Size, handler &CGH) {
    if (Size == 1 && MRWAcc != nullptr) {
      associateWithHandler(CGH);
      return *MRWAcc;
    }

    // Create a new output buffer and return an accessor to it.
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
  }

  /// If reduction is initialized with read-write accessor, which does not
  /// require initialization with identity value, then return user's read-write
  /// accessor. Otherwise, create 1-element global buffer initialized with
  /// identity value and return an accessor to that buffer.

  template <bool HasFastAtomics = (has_fast_atomics || has_atomic_add_float64)>
  std::enable_if_t<HasFastAtomics, rw_accessor_type>
  getReadWriteAccessorToInitializedMem(handler &CGH) {
    if (!is_usm && !initializeToIdentity())
      return *MRWAcc;

    auto RWReduVal = std::make_shared<T>(MIdentity);
    CGH.addReduction(RWReduVal);
    MOutBufPtr = std::make_shared<buffer<T, 1>>(RWReduVal.get(), range<1>(1));
    CGH.addReduction(MOutBufPtr);
    return createHandlerWiredReadWriteAccessor(CGH, *MOutBufPtr);
  }

  accessor<int, 1, access::mode::read_write, access::target::global_buffer,
           access::placeholder::false_t>
  getReadWriteAccessorToInitializedGroupsCounter(handler &CGH) {
    auto CounterMem = std::make_shared<int>(0);
    CGH.addReduction(CounterMem);
    auto CounterBuf = std::make_shared<buffer<int, 1>>(CounterMem.get(), 1);
    CGH.addReduction(CounterBuf);
    return {*CounterBuf, CGH};
  }

  bool hasUserDiscardWriteAccessor() { return MDWAcc != nullptr; }

  template <bool _IsUSM = IsUSM>
  std::enable_if_t<!_IsUSM, rw_accessor_type &> getUserReadWriteAccessor() {
    return *MRWAcc;
  }

  template <bool _IsUSM = IsUSM>
  std::enable_if_t<!_IsUSM, dw_accessor_type &> getUserDiscardWriteAccessor() {
    return *MDWAcc;
  }

  result_type *getUSMPointer() {
    assert(is_usm && "Unexpected call of getUSMPointer().");
    return MUSMPointer;
  }

  static inline result_type *getOutPointer(const rw_accessor_type &OutAcc) {
    return OutAcc.get_pointer().get();
  }

  static inline result_type *getOutPointer(result_type *OutPtr) {
    return OutPtr;
  }

  /// Returns the binary operation associated with the reduction.
  BinaryOperation getBinaryOperation() const { return MBinaryOp; }
  bool initializeToIdentity() const { return InitializeToIdentity; }

private:
  template <typename BufferT, access::placeholder IsPH = IsPlaceholder>
  std::enable_if_t<IsPH == access::placeholder::false_t, rw_accessor_type>
  createHandlerWiredReadWriteAccessor(handler &CGH, BufferT Buffer) {
    return {Buffer, CGH};
  }

  template <typename BufferT, access::placeholder IsPH = IsPlaceholder>
  std::enable_if_t<IsPH == access::placeholder::true_t, rw_accessor_type>
  createHandlerWiredReadWriteAccessor(handler &CGH, BufferT Buffer) {
    rw_accessor_type Acc(Buffer);
    CGH.require(Acc);
    return Acc;
  }

  /// Identity of the BinaryOperation.
  /// The result of BinaryOperation(X, MIdentity) is equal to X for any X.
  const T MIdentity;

  /// User's accessor to where the reduction must be written.
  std::shared_ptr<rw_accessor_type> MRWAcc;
  std::shared_ptr<dw_accessor_type> MDWAcc;

  std::shared_ptr<buffer<T, buffer_dim>> MOutBufPtr;

  /// USM pointer referencing the memory to where the result of the reduction
  /// must be written. Applicable/used only for USM reductions.
  T *MUSMPointer = nullptr;

  BinaryOperation MBinaryOp;

  bool InitializeToIdentity;
};

/// These are the forward declaration for the classes that help to create
/// names for additional kernels. It is used only when there are
/// more then 1 kernels in one parallel_for() implementing SYCL reduction.
template <typename T1, bool B1, bool B2, typename T2>
class __sycl_reduction_main_kernel;
template <typename T1, bool B1, bool B2, typename T2>
class __sycl_reduction_aux_kernel;

/// Helper structs to get additional kernel name types based on given
/// \c Name and additional template parameters helping to distinguish kernels.
/// If \c Name is undefined (is \c auto_name) leave it that way to take
/// advantage of unnamed kernels being named after their functor.
template <typename Name, typename Type, bool B1, bool B2, typename T3 = void>
struct get_reduction_main_kernel_name_t {
  using name = __sycl_reduction_main_kernel<Name, B1, B2, T3>;
};
template <typename Type, bool B1, bool B2, typename T3>
struct get_reduction_main_kernel_name_t<sycl::detail::auto_name, Type, B1, B2,
                                        T3> {
  using name = sycl::detail::auto_name;
};
template <typename Name, typename Type, bool B1, bool B2, typename T3>
struct get_reduction_aux_kernel_name_t {
  using name = __sycl_reduction_aux_kernel<Name, B1, B2, T3>;
};
template <typename Type, bool B1, bool B2, typename T3>
struct get_reduction_aux_kernel_name_t<sycl::detail::auto_name, Type, B1, B2,
                                       T3> {
  using name = sycl::detail::auto_name;
};

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

template <typename KernelName, typename KernelType, int Dims, class Reduction>
std::enable_if_t<Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const range<Dims> &Range,
               const nd_range<1> &NDRange, Reduction &Redu) {
  auto Out = Redu.getReadWriteAccessorToInitializedMem(CGH);
  auto GroupSum = Reduction::getReadWriteLocalAcc(1, CGH);
  using Name =
      typename get_reduction_main_kernel_name_t<KernelName, KernelType,
                                                Reduction::is_usm, false>::name;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    auto LID = NDId.get_local_id(0);
    if (LID == 0)
      GroupSum[0] = Reducer.getIdentity();
    sycl::detail::workGroupBarrier();
    Reducer.template atomic_combine<access::address_space::local_space>(
        &GroupSum[0]);

    sycl::detail::workGroupBarrier();
    if (LID == 0) {
      Reducer.MValue = GroupSum[0];
      Reducer.template atomic_combine(Reduction::getOutPointer(Out));
    }
  });
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
std::enable_if_t<!Reduction::has_fast_atomics && Reduction::has_fast_reduce>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const range<Dims> &Range,
               const nd_range<1> &NDRange, Reduction &Redu) {
  size_t WGSize = NDRange.get_local_range().size();
  size_t NWorkGroups = NDRange.get_group_range().size();

  bool IsUpdateOfUserVar = !Reduction::is_usm && !Redu.initializeToIdentity();
  auto PartialSums = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
  auto Out =
      (NWorkGroups == 1) ? PartialSums : Redu.getWriteAccForPartialReds(1, CGH);
  auto NWorkGroupsFinished =
      Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH);
  auto DoReducePartialSumsInLastWG =
      Reduction::template getReadWriteLocalAcc<int>(1, CGH);

  using Name =
      typename get_reduction_main_kernel_name_t<KernelName, KernelType,
                                                Reduction::is_usm, false>::name;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    typename Reduction::binary_operation BOp;
    auto Group = NDId.get_group();
    Reducer.MValue = reduce_over_group(Group, Reducer.MValue, BOp);

    size_t LID = NDId.get_local_id(0);
    if (LID == 0) {
      if (NWorkGroups == 1 && IsUpdateOfUserVar)
        Reducer.MValue = BOp(Reducer.MValue, *Reduction::getOutPointer(Out));
      // if NWorkGroups == 1, then PartialsSum and Out point to same memory.
      Reduction::getOutPointer(PartialSums)[NDId.get_group_linear_id()] =
          Reducer.MValue;

      auto NFinished =
          atomic_ref<int, memory_order::relaxed, memory_scope::device,
                     access::address_space::global_space>(
              NWorkGroupsFinished[0]);
      DoReducePartialSumsInLastWG[0] =
          ++NFinished == NWorkGroups && NWorkGroups > 1;
    }

    sycl::detail::workGroupBarrier();
    if (DoReducePartialSumsInLastWG[0]) {
      auto LocalSum = Reducer.getIdentity();
      for (size_t I = LID; I < NWorkGroups; I += WGSize)
        LocalSum = BOp(LocalSum, PartialSums[I]);
      Reducer.MValue = reduce_over_group(Group, LocalSum, BOp);

      if (LID == 0) {
        if (IsUpdateOfUserVar)
          Reducer.MValue = BOp(Reducer.MValue, *Reduction::getOutPointer(Out));
        Reduction::getOutPointer(Out)[0] = Reducer.MValue;
      }
    }
  });
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
std::enable_if_t<!Reduction::has_fast_atomics && !Reduction::has_fast_reduce>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const range<Dims> &Range,
               const nd_range<1> &NDRange, Reduction &Redu) {
  size_t WGSize = NDRange.get_local_range().size();
  size_t NWorkGroups = NDRange.get_group_range().size();

  bool IsUpdateOfUserVar = !Reduction::is_usm && !Redu.initializeToIdentity();
  auto PartialSums = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
  auto Out =
      (NWorkGroups == 1) ? PartialSums : Redu.getWriteAccForPartialReds(1, CGH);
  auto LocalReds = Reduction::getReadWriteLocalAcc(WGSize + 1, CGH);
  auto NWorkGroupsFinished =
      Redu.getReadWriteAccessorToInitializedGroupsCounter(CGH);
  auto DoReducePartialSumsInLastWG =
      Reduction::template getReadWriteLocalAcc<int>(1, CGH);

  auto Identity = Redu.getIdentity();
  auto BOp = Redu.getBinaryOperation();
  using Name =
      typename get_reduction_main_kernel_name_t<KernelName, KernelType,
                                                Reduction::is_usm, false>::name;
  CGH.parallel_for<Name>(NDRange, [=](nd_item<1> NDId) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer(Identity, BOp);
    reductionLoop(Range, Reducer, NDId, KernelFunc);

    // Copy the element to local memory to prepare it for tree-reduction.
    size_t LID = NDId.get_local_linear_id();
    LocalReds[LID] = Reducer.MValue;
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
        V = BOp(V, *Reduction::getOutPointer(Out));
      // if NWorkGroups == 1, then PartialsSum and Out point to same memory.
      Reduction::getOutPointer(PartialSums)[NDId.get_group_linear_id()] = V;

      auto NFinished =
          atomic_ref<int, memory_order::relaxed, memory_scope::device,
                     access::address_space::global_space>(
              NWorkGroupsFinished[0]);
      DoReducePartialSumsInLastWG[0] =
          ++NFinished == NWorkGroups && NWorkGroups > 1;
    }

    sycl::detail::workGroupBarrier();
    if (DoReducePartialSumsInLastWG[0]) {
      auto LocalSum = Identity;
      for (size_t I = LID; I < NWorkGroups; I += WGSize)
        LocalSum = BOp(LocalSum, Reduction::getOutPointer(PartialSums)[I]);

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
          V = BOp(V, *Reduction::getOutPointer(Out));
        Reduction::getOutPointer(Out)[0] = V;
      }
    }
  });
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
void reduCGFunc(handler &CGH, KernelType KernelFunc, const range<Dims> &Range,
                size_t MaxWGSize, uint32_t NumConcurrentWorkGroups,
                Reduction &Redu) {
  size_t NWorkItems = Range.size();
  size_t WGSize = std::min(NWorkItems, MaxWGSize);
  size_t NWorkGroups = NWorkItems / WGSize;
  if (NWorkItems % WGSize)
    NWorkGroups++;
  size_t MaxNWorkGroups = NumConcurrentWorkGroups;
  NWorkGroups = std::min(NWorkGroups, MaxNWorkGroups);
  size_t NDRItems = NWorkGroups * WGSize;
  nd_range<1> NDRange{range<1>{NDRItems}, range<1>{WGSize}};

  reduCGFuncImpl<KernelName>(CGH, KernelFunc, Range, NDRange, Redu);
}

/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups, then it calls fast SYCL atomic operations to update
/// the given reduction variable \p Out.
///
/// Briefly: calls user's lambda, ext::oneapi::reduce() + atomic, INT +
/// ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG>
enable_if_t<Reduction::has_fast_reduce && Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &, typename Reduction::rw_accessor_type Out) {
  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's function. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    typename Reduction::binary_operation BOp;
    Reducer.MValue = ext::oneapi::reduce(NDIt.get_group(), Reducer.MValue, BOp);
    if (NDIt.get_local_linear_id() == 0)
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
  });
}

/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups, then it calls fast SYCL atomic operations to update
/// user's reduction variable.
///
/// Briefly: calls user's lambda, tree-reduction + atomic, INT + AND/OR/XOR.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG>
enable_if_t<!Reduction::has_fast_reduce && Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &, typename Reduction::rw_accessor_type Out) {
  size_t WGSize = Range.get_local_range().size();

  // Use local memory to reduce elements in work-groups into zero-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch reduce elements that could
  // otherwise be lost in the tree-reduction algorithm used in the kernel.
  size_t NLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NLocalElements, CGH);

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();

    // Copy the element to local memory to prepare it for tree-reduction.
    LocalReds[LID] = Reducer.MValue;
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
      Reducer.MValue =
          IsPow2WG ? LocalReds[0] : BOp(LocalReds[0], LocalReds[WGSize]);
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
    }
  });
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<Reduction::has_fast_atomics>
reduCGFunc(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
           Reduction &Redu) {

  size_t WGSize = Range.get_local_range().size();

  // User's initialized read-write accessor is re-used here if
  // initialize_to_identity is not set (i.e. if user's variable is initialized).
  // Otherwise, a new buffer is initialized with identity value and a new
  // read-write accessor to that buffer is created. That is done because
  // atomic operations update some initialized memory.
  // User's USM pointer is not re-used even when initialize_to_identity is not
  // set because it does not worth the creation of an additional variant of
  // a user's kernel for that case.
  auto Out = Redu.getReadWriteAccessorToInitializedMem(CGH);

  // If the work group size is not pow of 2, then the kernel runs some
  // additional code and checks in it.
  // If the reduction has fast reduce then the kernel does not care if the work
  // group size is pow of 2 or not, assume true for such cases.
  bool IsPow2WG = Reduction::has_fast_reduce || ((WGSize & (WGSize - 1)) == 0);
  if (IsPow2WG)
    reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, true>(
        CGH, KernelFunc, Range, Redu, Out);
  else
    reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, false>(
        CGH, KernelFunc, Range, Redu, Out);
}

/// Implements a command group function that enqueues a kernel that
/// calls user's lambda function and does one iteration of reduction
/// of elements in each of work-groups.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: user's lambda, ext::oneapi::reduce(), FP + ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG>
enable_if_t<Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &Redu, typename Reduction::rw_accessor_type Out) {

  size_t NWorkGroups = Range.get_group_range().size();
  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    // Compute the partial sum/reduction for the work-group.
    size_t WGID = NDIt.get_group_linear_id();
    typename Reduction::result_type PSum = Reducer.MValue;
    typename Reduction::binary_operation BOp;
    PSum = ext::oneapi::reduce(NDIt.get_group(), PSum, BOp);
    if (NDIt.get_local_linear_id() == 0) {
      if (IsUpdateOfUserVar)
        PSum = BOp(*(Reduction::getOutPointer(Out)), PSum);
      Reduction::getOutPointer(Out)[WGID] = PSum;
    }
  });
}

/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function \param KernelFunc and does one iteration of
/// reduction of elements in each of work-groups.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups. At the end of each work-group the partial sum is written
/// to a global buffer.
///
/// Briefly: user's lambda, tree-reduction, CUSTOM types/ops.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG>
enable_if_t<!Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &Redu, typename Reduction::rw_accessor_type Out) {
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
  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG>::name;
  auto BOp = Redu.getBinaryOperation();
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer(ReduIdentity, BOp);
    KernelFunc(NDIt, Reducer);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    // Copy the element to local memory to prepare it for tree-reduction.
    LocalReds[LID] = Reducer.MValue;
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
      Reduction::getOutPointer(Out)[GrID] = PSum;
    }
  });
}

template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<!Reduction::has_fast_atomics>
reduCGFunc(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
           Reduction &Redu) {
  size_t WGSize = Range.get_local_range().size();
  size_t NWorkGroups = Range.get_group_range().size();

  // If the work group size is not pow of 2, then the kernel runs some
  // additional code and checks in it.
  // If the reduction has fast reduce then the kernel does not care if the work
  // group size is pow of 2 or not, assume true for such cases.
  bool IsPow2WG = Reduction::has_fast_reduce || ((WGSize & (WGSize - 1)) == 0);

  auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
  if (IsPow2WG)
    reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, true>(
        CGH, KernelFunc, Range, Redu, Out);
  else
    reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, false>(
        CGH, KernelFunc, Range, Redu, Out);
}

/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// This version uses ext::oneapi::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: aux kernel, ext::oneapi::reduce(), reproducible results, FP +
/// ADD/MIN/MAX
template <typename KernelName, typename KernelType, bool UniformWG,
          class Reduction, typename InputT, typename OutputT>
enable_if_t<Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduAuxCGFuncImpl(handler &CGH, size_t NWorkItems, size_t NWorkGroups,
                  size_t WGSize, Reduction &Redu, InputT In, OutputT Out) {
  using Name = typename get_reduction_aux_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, UniformWG, OutputT>::name;
  bool IsUpdateOfUserVar =
      !Reduction::is_usm && !Redu.initializeToIdentity() && NWorkGroups == 1;
  range<1> GlobalRange = {UniformWG ? NWorkItems : NWorkGroups * WGSize};
  nd_range<1> Range{GlobalRange, range<1>(WGSize)};
  CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
    typename Reduction::binary_operation BOp;
    size_t WGID = NDIt.get_group_linear_id();
    size_t GID = NDIt.get_global_linear_id();
    typename Reduction::result_type PSum =
        (UniformWG || (GID < NWorkItems))
            ? In[GID]
            : Reduction::reducer_type::getIdentity();
    PSum = ext::oneapi::reduce(NDIt.get_group(), PSum, BOp);
    if (NDIt.get_local_linear_id() == 0) {
      if (IsUpdateOfUserVar)
        PSum = BOp(*(Reduction::getOutPointer(Out)), PSum);
      Reduction::getOutPointer(Out)[WGID] = PSum;
    }
  });
}

/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// This version uses tree-reduction algorithm to reduce elements in each
/// of work-groups. At the end of each work-group the partial sum is written
/// to a global buffer.
///
/// Briefly: aux kernel, tree-reduction, CUSTOM types/ops.
template <typename KernelName, typename KernelType, bool UniformPow2WG,
          class Reduction, typename InputT, typename OutputT>
enable_if_t<!Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduAuxCGFuncImpl(handler &CGH, size_t NWorkItems, size_t NWorkGroups,
                  size_t WGSize, Reduction &Redu, InputT In, OutputT Out) {
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
  using Name = typename get_reduction_aux_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, UniformPow2WG, OutputT>::name;
  range<1> GlobalRange = {UniformPow2WG ? NWorkItems : NWorkGroups * WGSize};
  nd_range<1> Range{GlobalRange, range<1>(WGSize)};
  CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    size_t GID = NDIt.get_global_linear_id();

    // Copy the element to local memory to prepare it for tree-reduction.
    LocalReds[LID] =
        (UniformPow2WG || GID < NWorkItems) ? In[GID] : ReduIdentity;
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
      Reduction::getOutPointer(Out)[GrID] = PSum;
    }
  });
}

/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// At the end of each work-group the partial sum is written to a global buffer.
/// The function returns the number of the newly generated partial sums.
template <typename KernelName, typename KernelType, class Reduction>
enable_if_t<!Reduction::has_fast_atomics, size_t>
reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
              Reduction &Redu) {

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
  auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
  if (HasUniformWG)
    reduAuxCGFuncImpl<KernelName, KernelType, true>(
        CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);
  else
    reduAuxCGFuncImpl<KernelName, KernelType, false>(
        CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);
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
  Redu.associateWithHandler(CGH);
  if (Redu.hasUserDiscardWriteAccessor())
    CGH.copy(InAcc, Redu.getUserDiscardWriteAccessor());
  else
    CGH.copy(InAcc, Redu.getUserReadWriteAccessor());
}

// This method is used for implementation of parallel_for accepting 1 reduction.
// TODO: remove this method when everything is switched to general algorithm
// implementing arbitrary number of reductions in parallel_for().
/// Copies the final reduction result kept in read-write accessor to user's
/// USM memory.
template <typename KernelName, class Reduction>
std::enable_if_t<Reduction::is_usm>
reduSaveFinalResultToUserMem(handler &CGH, Reduction &Redu) {
  auto InAcc = Redu.getReadAccToPreviousPartialReds(CGH);
  auto UserVarPtr = Redu.getUSMPointer();
  bool IsUpdateOfUserVar = !Redu.initializeToIdentity();
  auto BOp = Redu.getBinaryOperation();
  CGH.single_task<KernelName>([=] {
    if (IsUpdateOfUserVar)
      *UserVarPtr = BOp(*UserVarPtr, *(InAcc.get_pointer()));
    else
      *UserVarPtr = *(InAcc.get_pointer());
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
          NWorkGroups, CGH)...);
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

template <bool Pow2WG, typename... LocalAccT, typename... ReducerT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t WGSize,
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

template <bool UniformPow2WG, typename... LocalAccT, typename... InputAccT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t GID, size_t NWorkItems, size_t WGSize,
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

template <bool Pow2WG, bool IsOneWG, typename... Reductions,
          typename... OutAccT, typename... LocalAccT, typename... BOPsT,
          typename... Ts, size_t... Is>
void writeReduSumsToOutAccs(
    size_t OutAccIndex, size_t WGSize, std::tuple<Reductions...> *,
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
  template <typename T> struct Func { static constexpr bool value = false; };
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

template <typename KernelName, bool Pow2WG, bool IsOneWG, typename KernelType,
          int Dims, typename... Reductions, size_t... Is>
void reduCGFuncImpl(handler &CGH, KernelType KernelFunc,
                    const nd_range<Dims> &Range,
                    std::tuple<Reductions...> &ReduTuple,
                    std::index_sequence<Is...> ReduIndices) {

  size_t WGSize = Range.get_local_range().size();
  size_t LocalAccSize = WGSize + (Pow2WG ? 0 : 1);
  auto LocalAccsTuple =
      createReduLocalAccs<Reductions...>(LocalAccSize, CGH, ReduIndices);

  size_t NWorkGroups = IsOneWG ? 1 : Range.get_group_range().size();
  auto OutAccsTuple =
      createReduOutAccs<IsOneWG>(NWorkGroups, CGH, ReduTuple, ReduIndices);
  auto IdentitiesTuple = getReduIdentities(ReduTuple, ReduIndices);
  auto BOPsTuple = getReduBOPs(ReduTuple, ReduIndices);
  auto InitToIdentityProps =
      getInitToIdentityProperties(ReduTuple, ReduIndices);

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Pow2WG, IsOneWG, decltype(OutAccsTuple)>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    auto ReduIndices = std::index_sequence_for<Reductions...>();
    auto ReducersTuple =
        createReducers<Reductions...>(IdentitiesTuple, BOPsTuple, ReduIndices);
    // The .MValue field of each of the elements in ReducersTuple
    // gets initialized in this call.
    callReduUserKernelFunc(KernelFunc, NDIt, ReducersTuple, ReduIndices);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    initReduLocalAccs<Pow2WG>(LID, WGSize, LocalAccsTuple, ReducersTuple,
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
      writeReduSumsToOutAccs<Pow2WG, IsOneWG>(
          GrID, WGSize, (std::tuple<Reductions...> *)nullptr, OutAccsTuple,
          LocalAccsTuple, BOPsTuple, IdentitiesTuple, InitToIdentityProps,
          ReduIndices);
    }
  });
}

template <typename KernelName, typename KernelType, int Dims,
          typename... Reductions, size_t... Is>
void reduCGFunc(handler &CGH, KernelType KernelFunc,
                const nd_range<Dims> &Range,
                std::tuple<Reductions...> &ReduTuple,
                std::index_sequence<Is...> ReduIndices) {
  size_t WGSize = Range.get_local_range().size();
  size_t NWorkGroups = Range.get_group_range().size();
  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;
  if (NWorkGroups == 1) {
    // TODO: consider having only one variant of kernel instead of two here.
    // Having two kernels, where one is just slighly more efficient than
    // another, and only for the purpose of running 1 work-group may be too
    // expensive.
    if (Pow2WG)
      reduCGFuncImpl<KernelName, true, true>(CGH, KernelFunc, Range, ReduTuple,
                                             ReduIndices);
    else
      reduCGFuncImpl<KernelName, false, true>(CGH, KernelFunc, Range, ReduTuple,
                                              ReduIndices);
  } else {
    if (Pow2WG)
      reduCGFuncImpl<KernelName, true, false>(CGH, KernelFunc, Range, ReduTuple,
                                              ReduIndices);
    else
      reduCGFuncImpl<KernelName, false, false>(CGH, KernelFunc, Range,
                                               ReduTuple, ReduIndices);
  }
}

// Specialization for devices with the atomic64 aspect, which guarantees 64 (and
// temporarily 32) bit floating point support for atomic add.
// TODO 32 bit floating point atomics are eventually expected to be supported by
// the has_fast_atomics specialization. Corresponding changes to
// IsReduOptForAtomic64Add, as prescribed in its documentation, should then also
// be made.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
std::enable_if_t<Reduction::has_atomic_add_float64>
reduCGFuncImplAtomic64(handler &CGH, KernelType KernelFunc,
                       const nd_range<Dims> &Range, Reduction &,
                       typename Reduction::rw_accessor_type Out) {
  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm,
      Reduction::has_atomic_add_float64,
      typename Reduction::rw_accessor_type>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's function. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    typename Reduction::binary_operation BOp;
    Reducer.MValue = reduce_over_group(NDIt.get_group(), Reducer.MValue, BOp);
    if (NDIt.get_local_linear_id() == 0) {
      Reducer.atomic_combine(Reduction::getOutPointer(Out));
    }
  });
}

// Specialization for devices with the atomic64 aspect, which guarantees 64 (and
// temporarily 32) bit floating point support for atomic add.
// TODO 32 bit floating point atomics are eventually expected to be supported by
// the has_fast_atomics specialization. Corresponding changes to
// IsReduOptForAtomic64Add, as prescribed in its documentation, should then also
// be made.
template <typename KernelName, typename KernelType, int Dims, class Reduction>
enable_if_t<Reduction::has_atomic_add_float64>
reduCGFuncAtomic64(handler &CGH, KernelType KernelFunc,
                   const nd_range<Dims> &Range, Reduction &Redu) {

  auto Out = Redu.getReadWriteAccessorToInitializedMem(CGH);
  reduCGFuncImplAtomic64<KernelName, KernelType, Dims, Reduction>(
      CGH, KernelFunc, Range, Redu, Out);
}

inline void associateReduAccsWithHandlerHelper(handler &) {}

template <typename ReductionT>
void associateReduAccsWithHandlerHelper(handler &CGH, ReductionT &Redu) {
  Redu.associateWithHandler(CGH);
}

template <typename ReductionT, typename... RestT,
          enable_if_t<(sizeof...(RestT) > 0), int> Z = 0>
void associateReduAccsWithHandlerHelper(handler &CGH, ReductionT &Redu,
                                        RestT &... Rest) {
  Redu.associateWithHandler(CGH);
  associateReduAccsWithHandlerHelper(CGH, Rest...);
}

template <typename... Reductions, size_t... Is>
void associateReduAccsWithHandler(handler &CGH,
                                  std::tuple<Reductions...> &ReduTuple,
                                  std::index_sequence<Is...>) {
  associateReduAccsWithHandlerHelper(CGH, std::get<Is>(ReduTuple)...);
}

template <typename KernelName, typename KernelType, bool UniformPow2WG,
          bool IsOneWG, typename... Reductions, size_t... Is>
void reduAuxCGFuncImpl(handler &CGH, size_t NWorkItems, size_t NWorkGroups,
                       size_t WGSize, std::tuple<Reductions...> &ReduTuple,
                       std::index_sequence<Is...> ReduIndices) {
  // The last kernel DOES write to user's accessor passed to reduction.
  // Associate it with handler manually.
  std::conditional_t<IsOneWG, IsNonUsmReductionPredicate,
                     EmptyReductionPredicate>
      Predicate;
  auto AccReduIndices = filterSequence<Reductions...>(Predicate, ReduIndices);
  associateReduAccsWithHandler(CGH, ReduTuple, AccReduIndices);

  size_t LocalAccSize = WGSize + (UniformPow2WG ? 0 : 1);
  auto LocalAccsTuple =
      createReduLocalAccs<Reductions...>(LocalAccSize, CGH, ReduIndices);
  auto InAccsTuple =
      getReadAccsToPreviousPartialReds(CGH, ReduTuple, ReduIndices);
  auto OutAccsTuple =
      createReduOutAccs<IsOneWG>(NWorkGroups, CGH, ReduTuple, ReduIndices);
  auto IdentitiesTuple = getReduIdentities(ReduTuple, ReduIndices);
  auto BOPsTuple = getReduBOPs(ReduTuple, ReduIndices);
  auto InitToIdentityProps =
      getInitToIdentityProperties(ReduTuple, ReduIndices);

  using Name =
      typename get_reduction_aux_kernel_name_t<KernelName, KernelType,
                                               UniformPow2WG, IsOneWG,
                                               decltype(OutAccsTuple)>::name;
  range<1> GlobalRange = {UniformPow2WG ? NWorkItems : NWorkGroups * WGSize};
  nd_range<1> Range{GlobalRange, range<1>(WGSize)};
  CGH.parallel_for<Name>(Range, [=](nd_item<1> NDIt) {
    auto ReduIndices = std::index_sequence_for<Reductions...>();
    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    size_t GID = NDIt.get_global_linear_id();
    initReduLocalAccs<UniformPow2WG>(LID, GID, NWorkItems, WGSize,
                                     LocalAccsTuple, InAccsTuple,
                                     IdentitiesTuple, ReduIndices);
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
      writeReduSumsToOutAccs<UniformPow2WG, IsOneWG>(
          GrID, WGSize, (std::tuple<Reductions...> *)nullptr, OutAccsTuple,
          LocalAccsTuple, BOPsTuple, IdentitiesTuple, InitToIdentityProps,
          ReduIndices);
    }
  });
}

template <typename KernelName, typename KernelType, typename... Reductions,
          size_t... Is>
size_t reduAuxCGFunc(handler &CGH, size_t NWorkItems, size_t MaxWGSize,
                     std::tuple<Reductions...> &ReduTuple,
                     std::index_sequence<Is...> ReduIndices) {
  size_t NWorkGroups;
  size_t WGSize = reduComputeWGSize(NWorkItems, MaxWGSize, NWorkGroups);

  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;
  bool HasUniformWG = Pow2WG && (NWorkGroups * WGSize == NWorkItems);
  if (NWorkGroups == 1) {
    if (HasUniformWG)
      reduAuxCGFuncImpl<KernelName, KernelType, true, true>(
          CGH, NWorkItems, NWorkGroups, WGSize, ReduTuple, ReduIndices);
    else
      reduAuxCGFuncImpl<KernelName, KernelType, false, true>(
          CGH, NWorkItems, NWorkGroups, WGSize, ReduTuple, ReduIndices);
  } else {
    if (HasUniformWG)
      reduAuxCGFuncImpl<KernelName, KernelType, true, false>(
          CGH, NWorkItems, NWorkGroups, WGSize, ReduTuple, ReduIndices);
    else
      reduAuxCGFuncImpl<KernelName, KernelType, false, false>(
          CGH, NWorkItems, NWorkGroups, WGSize, ReduTuple, ReduIndices);
  }
  return NWorkGroups;
}

inline void
reduSaveFinalResultToUserMemHelper(std::vector<event> &,
                                   std::shared_ptr<detail::queue_impl>, bool) {}

template <typename Reduction, typename... RestT>
std::enable_if_t<Reduction::is_usm>
reduSaveFinalResultToUserMemHelper(std::vector<event> &Events,
                                   std::shared_ptr<detail::queue_impl> Queue,
                                   bool IsHost, Reduction &, RestT... Rest) {
  // Reductions initialized with USM pointer currently do not require copying
  // because the last kernel write directly to USM memory.
  reduSaveFinalResultToUserMemHelper(Events, Queue, IsHost, Rest...);
}

template <typename Reduction, typename... RestT>
std::enable_if_t<!Reduction::is_usm> reduSaveFinalResultToUserMemHelper(
    std::vector<event> &Events, std::shared_ptr<detail::queue_impl> Queue,
    bool IsHost, Reduction &Redu, RestT... Rest) {
  if (Redu.hasUserDiscardWriteAccessor()) {
    handler CopyHandler(Queue, IsHost);
    auto InAcc = Redu.getReadAccToPreviousPartialReds(CopyHandler);
    auto OutAcc = Redu.getUserDiscardWriteAccessor();
    Redu.associateWithHandler(CopyHandler);
    if (!Events.empty())
      CopyHandler.depends_on(Events.back());
    CopyHandler.copy(InAcc, OutAcc);
    event CopyEvent = CopyHandler.finalize();
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
detail::reduction_impl<T, BinaryOperation, Dims, false, IsPH>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          const T &Identity, BinaryOperation BOp) {
  return {Acc, Identity, BOp};
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc and the binary operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH>
std::enable_if_t<detail::IsKnownIdentityOp<T, BinaryOperation>::value,
                 detail::reduction_impl<T, BinaryOperation, Dims, false, IsPH>>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          BinaryOperation) {
  return {Acc};
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 3 arguments: the reference to the reduction variable to where
/// the computed reduction must be stored \param VarPtr, identity value
/// \param Identity, and the binary operation used in the reduction.
template <typename T, class BinaryOperation>
detail::reduction_impl<T, BinaryOperation, 1, true>
reduction(T *VarPtr, const T &Identity, BinaryOperation BOp) {
  return {VarPtr, Identity, BOp};
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the reference to the reduction variable, to where
/// the computed reduction must be stored \param VarPtr, and the binary
/// operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation>
std::enable_if_t<detail::IsKnownIdentityOp<T, BinaryOperation>::value,
                 detail::reduction_impl<T, BinaryOperation, 1, true>>
reduction(T *VarPtr, BinaryOperation) {
  return {VarPtr};
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

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
  namespace detail {
  using cl::sycl::detail::queue_impl;
  __SYCL_EXPORT size_t reduGetMaxWGSize(shared_ptr_class<queue_impl> Queue,
                                        size_t LocalMemBytesPerWorkItem);
  __SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                         size_t &NWorkGroups);
  } // namespace detail
} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
