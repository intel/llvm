//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "CL/sycl/ONEAPI/accessor_property_list.hpp"
#include <CL/sycl/ONEAPI/group_algorithm.hpp>
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/kernel.hpp>

#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ONEAPI {

namespace detail {

using cl::sycl::detail::queue_impl;

__SYCL_EXPORT size_t reduGetMaxWGSize(shared_ptr_class<queue_impl> Queue,
                                      size_t LocalMemBytesPerWorkItem);
__SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                       size_t &NWorkGroups);

using cl::sycl::detail::bool_constant;
using cl::sycl::detail::enable_if_t;
using cl::sycl::detail::is_geninteger16bit;
using cl::sycl::detail::is_geninteger32bit;
using cl::sycl::detail::is_geninteger64bit;
using cl::sycl::detail::is_geninteger8bit;
using cl::sycl::detail::remove_AS;

template <typename T, class BinaryOperation>
using IsReduPlus = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::plus<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::plus<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduMultiplies = detail::bool_constant<
    std::is_same<BinaryOperation, std::multiplies<T>>::value ||
    std::is_same<BinaryOperation, std::multiplies<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduMinimum = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::minimum<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::minimum<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduMaximum = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::maximum<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::maximum<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduBitOR = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::bit_or<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::bit_or<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduBitXOR = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::bit_xor<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::bit_xor<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduBitAND = detail::bool_constant<
    std::is_same<BinaryOperation, ONEAPI::bit_and<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::bit_and<void>>::value>;

template <typename T, class BinaryOperation>
using IsReduOptForFastAtomicFetch =
    detail::bool_constant<(is_geninteger32bit<T>::value ||
                           is_geninteger64bit<T>::value) &&
                          (IsReduPlus<T, BinaryOperation>::value ||
                           IsReduMinimum<T, BinaryOperation>::value ||
                           IsReduMaximum<T, BinaryOperation>::value ||
                           IsReduBitOR<T, BinaryOperation>::value ||
                           IsReduBitXOR<T, BinaryOperation>::value ||
                           IsReduBitAND<T, BinaryOperation>::value)>;

template <typename T, class BinaryOperation>
using IsReduOptForFastReduce = detail::bool_constant<
    (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, half>::value || std::is_same<T, float>::value ||
     std::is_same<T, double>::value) &&
    (IsReduPlus<T, BinaryOperation>::value ||
     IsReduMinimum<T, BinaryOperation>::value ||
     IsReduMaximum<T, BinaryOperation>::value)>;

// Identity = 0
template <typename T, class BinaryOperation>
using IsZeroIdentityOp = bool_constant<
    ((is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
      is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
     (IsReduPlus<T, BinaryOperation>::value ||
      IsReduBitOR<T, BinaryOperation>::value ||
      IsReduBitXOR<T, BinaryOperation>::value)) ||
    ((std::is_same<T, half>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value) &&
     IsReduPlus<T, BinaryOperation>::value)>;

// Identity = 1
template <typename T, class BinaryOperation>
using IsOneIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, half>::value || std::is_same<T, float>::value ||
     std::is_same<T, double>::value) &&
    IsReduMultiplies<T, BinaryOperation>::value>;

// Identity = ~0
template <typename T, class BinaryOperation>
using IsOnesIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
    IsReduBitAND<T, BinaryOperation>::value>;

// Identity = <max possible value>
template <typename T, class BinaryOperation>
using IsMinimumIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, half>::value || std::is_same<T, float>::value ||
     std::is_same<T, double>::value) &&
    IsReduMinimum<T, BinaryOperation>::value>;

// Identity = <min possible value>
template <typename T, class BinaryOperation>
using IsMaximumIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, half>::value || std::is_same<T, float>::value ||
     std::is_same<T, double>::value) &&
    IsReduMaximum<T, BinaryOperation>::value>;

template <typename T, class BinaryOperation>
using IsKnownIdentityOp =
    bool_constant<IsZeroIdentityOp<T, BinaryOperation>::value ||
                  IsOneIdentityOp<T, BinaryOperation>::value ||
                  IsOnesIdentityOp<T, BinaryOperation>::value ||
                  IsMinimumIdentityOp<T, BinaryOperation>::value ||
                  IsMaximumIdentityOp<T, BinaryOperation>::value>;

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
/// For example, it is known that 0 is identity for ONEAPI::plus operations
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

  /// Returns zero as identity for ADD, OR, XOR operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsZeroIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return 0;
  }

  /// Returns one as identify for MULTIPLY operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsOneIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return 1;
  }

  /// Returns bit image consisting of all ones as identity for AND operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsOnesIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return ~static_cast<_T>(0);
  }

  /// Returns maximal possible value as identity for MIN operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsMinimumIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return std::numeric_limits<_T>::has_infinity
               ? std::numeric_limits<_T>::infinity()
               : (std::numeric_limits<_T>::max)();
  }

  /// Returns minimal possible value as identity for MAX operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsMaximumIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return std::numeric_limits<_T>::has_infinity
               ? static_cast<_T>(-std::numeric_limits<_T>::infinity())
               : std::numeric_limits<_T>::lowest();
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  IsReduPlus<T, BinaryOperation>::value,
              reducer &>
  operator+=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  IsReduMultiplies<T, BinaryOperation>::value,
              reducer &>
  operator*=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  IsReduBitOR<T, BinaryOperation>::value,
              reducer &>
  operator|=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  IsReduBitXOR<T, BinaryOperation>::value,
              reducer &>
  operator^=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  IsReduBitAND<T, BinaryOperation>::value,
              reducer &>
  operator&=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  /// Atomic ADD operation: *ReduVarPtr += MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduPlus<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_add(MValue);
  }

  /// Atomic BITWISE OR operation: *ReduVarPtr |= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduBitOR<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_or(MValue);
  }

  /// Atomic BITWISE XOR operation: *ReduVarPtr ^= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduBitXOR<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_xor(MValue);
  }

  /// Atomic BITWISE AND operation: *ReduVarPtr &= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduBitAND<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_and(MValue);
  }

  /// Atomic MIN operation: *ReduVarPtr = ONEAPI::minimum(*ReduVarPtr, MValue);
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduMinimum<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_min(MValue);
  }

  /// Atomic MAX operation: *ReduVarPtr = ONEAPI::maximum(*ReduVarPtr, MValue);
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              IsReduMaximum<T, _BinaryOperation>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_max(MValue);
  }

  T MValue;
};

/// Base non-template class which is a base class for all reduction
/// implementation classes. It is needed to detect the reduction classes.
class reduction_impl_base {};

/// Predicate returning true if and only if 'FirstT' is a reduction class and
/// all types except the last one from 'RestT' are reductions as well.
template <typename FirstT, typename... RestT>
struct are_all_but_last_reductions {
  static constexpr bool value =
      std::is_base_of<reduction_impl_base, FirstT>::value &&
      are_all_but_last_reductions<RestT...>::value;
};

/// Helper specialization of are_all_but_last_reductions for one element only.
/// Returns true if the last and only typename is not a reduction.
template <typename T> struct are_all_but_last_reductions<T> {
  static constexpr bool value = !std::is_base_of<reduction_impl_base, T>::value;
};

/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, bool IsUSM,
          access::mode AccMode = access::mode::read_write,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class reduction_impl : private reduction_impl_base {
public:
  using reducer_type = reducer<T, BinaryOperation>;
  using result_type = T;
  using binary_operation = BinaryOperation;
  using accessor_type =
      accessor<T, Dims, AccMode, access::target::global_buffer, IsPlaceholder,
               ONEAPI::accessor_property_list<>>;
  using rw_accessor_type =
      accessor<T, Dims, access::mode::read_write, access::target::global_buffer,
               IsPlaceholder, ONEAPI::accessor_property_list<>>;
  static constexpr access::mode accessor_mode = AccMode;
  static constexpr int accessor_dim = Dims;
  static constexpr int buffer_dim = (Dims == 0) ? 1 : Dims;
  using local_accessor_type =
      accessor<T, buffer_dim, access::mode::read_write, access::target::local>;

  static constexpr bool has_fast_atomics =
      IsReduOptForFastAtomicFetch<T, BinaryOperation>::value;
  static constexpr bool has_fast_reduce =
      IsReduOptForFastReduce<T, BinaryOperation>::value;
  static constexpr bool is_usm = IsUSM;
  static constexpr bool is_placeholder =
      (IsPlaceholder == access::placeholder::true_t);

  // Only scalar (i.e. 0-dim and 1-dim with 1 element) reductions supported now.
  // TODO: suport (Dims > 1) accessors/reductions.
  // TODO: support true 1-Dimensional accessors/reductions (get_count() > 1).
  // (get_count() == 1) is checked in the constructor of reduction_impl.
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

  /// Constructs reduction_impl when the identity value is statically known.
  // Note that aliasing constructor was used to initialize MAcc to avoid
  // destruction of the object referenced by the parameter Acc.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc)
      : MAcc(shared_ptr_class<accessor_type>(shared_ptr_class<accessor_type>{},
                                             &Acc)),
        MIdentity(getIdentity()) {
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
  }

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  // Note that aliasing constructor was used to initialize MAcc to avoid
  // destruction of the object referenced by the parameter Acc.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc, const T &Identity, BinaryOperation)
      : MAcc(shared_ptr_class<accessor_type>(shared_ptr_class<accessor_type>{},
                                             &Acc)),
        MIdentity(getIdentity()) {
    (void)Identity;
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
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
  // Note that aliasing constructor was used to initialize MAcc to avoid
  // destruction of the object referenced by the parameter Acc.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<!IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc, const T &Identity, BinaryOperation BOp)
      : MAcc(shared_ptr_class<accessor_type>(shared_ptr_class<accessor_type>{},
                                             &Acc)),
        MIdentity(Identity), MBinaryOp(BOp) {
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
  }

  /// Constructs reduction_impl when the identity value is statically known.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr) : MIdentity(getIdentity()), MUSMPointer(VarPtr) {}

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  /// The \param VarPtr is a USM pointer to memory, to where the computed
  /// reduction value is added using BinaryOperation, i.e. it is expected that
  /// the memory is pre-initialized with some meaningful value.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr, const T &Identity, BinaryOperation)
      : MIdentity(Identity), MUSMPointer(VarPtr) {
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
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<!IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(T *VarPtr, const T &Identity, BinaryOperation BOp)
      : MIdentity(Identity), MUSMPointer(VarPtr), MBinaryOp(BOp) {}

  /// Associates reduction accessor with the given handler and saves reduction
  /// buffer so that it is alive until the command group finishes the work.
  void associateWithHandler(handler &CGH) {
#ifndef __SYCL_DEVICE_ONLY__
    CGH.associateWithHandler(MAcc.get(), access::target::global_buffer);
#else
    (void)CGH;
#endif
  }

  static local_accessor_type getReadWriteLocalAcc(size_t Size, handler &CGH) {
    return local_accessor_type(Size, CGH);
  }

  accessor<T, buffer_dim, access::mode::read>
  getReadAccToPreviousPartialReds(handler &CGH) const {
    CGH.addReduction(MOutBufPtr);
    return accessor<T, buffer_dim, access::mode::read>(*MOutBufPtr, CGH);
  }

  /// Returns user's USM pointer passed to reduction for editing.
  template <bool IsOneWG, bool _IsUSM = is_usm>
  std::enable_if_t<IsOneWG && _IsUSM, result_type *>
  getWriteMemForPartialReds(size_t, handler &) {
    return getUSMPointer();
  }

  /// Returns user's accessor passed to reduction for editing.
  template <bool IsOneWG, bool _IsUSM = is_usm>
  std::enable_if_t<IsOneWG && !_IsUSM, accessor_type>
  getWriteMemForPartialReds(size_t, handler &) {
    return *MAcc;
  }

  /// Constructs a new temporary buffer to hold partial sums and returns
  /// the accessor that that buffer.
  template <bool IsOneWG>
  std::enable_if_t<!IsOneWG, accessor_type>
  getWriteMemForPartialReds(size_t Size, handler &CGH) {
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return accessor_type(*MOutBufPtr, CGH);
  }

  template <access::placeholder _IsPlaceholder = IsPlaceholder>
  enable_if_t<_IsPlaceholder == access::placeholder::false_t, accessor_type>
  getWriteAccForPartialReds(size_t Size, handler &CGH) {
    if (Size == 1)
      return *MAcc;

    // Create a new output buffer and return an accessor to it.
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return accessor_type(*MOutBufPtr, CGH);
  }

  template <access::placeholder _IsPlaceholder = IsPlaceholder>
  enable_if_t<_IsPlaceholder == access::placeholder::true_t, accessor_type>
  getWriteAccForPartialReds(size_t Size, handler &CGH) {
    if (Size == 1)
      return *MAcc;

    // Create a new output buffer and return an accessor to it.
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    accessor_type NewAcc(*MOutBufPtr);
    CGH.addReduction(MOutBufPtr);
    CGH.require(NewAcc);
    return NewAcc;
  }

  /// Creates 1-element global buffer initialized with identity value and
  /// returns an accessor to that buffer.
  accessor<T, Dims, access::mode::read_write, access::target::global_buffer>
  getReadWriteScalarAcc(handler &CGH) const {
    auto RWReduVal = std::make_shared<T>(MIdentity);
    CGH.addReduction(RWReduVal);
    auto RWReduBuf =
        std::make_shared<buffer<T, 1>>(RWReduVal.get(), range<1>(1));
    CGH.addReduction(RWReduBuf);
    return accessor<T, Dims, access::mode::read_write,
                    access::target::global_buffer>(*RWReduBuf, CGH);
  }

  accessor_type &getUserAccessor() { return *MAcc; }

  T *getUSMPointer() {
    assert(is_usm && "Unexpected call of getUSMPointer().");
    return MUSMPointer;
  }

  template <typename AccT>
  enable_if_t<std::is_same<AccT, rw_accessor_type>::value ||
                  std::is_same<AccT, accessor_type>::value,
              result_type *> static inline getOutPointer(const AccT &OutAcc) {
    return OutAcc.get_pointer().get();
  }

  template <typename ResT>
  enable_if_t<std::is_same<ResT, result_type>::value,
              result_type *> static inline getOutPointer(ResT *OutPtr) {
    return OutPtr;
  }

  /// Returns the binary operation associated with the reduction.
  BinaryOperation getBinaryOperation() const { return MBinaryOp; }

private:
  /// Identity of the BinaryOperation.
  /// The result of BinaryOperation(X, MIdentity) is equal to X for any X.
  const T MIdentity;

  /// User's accessor to where the reduction must be written.
  shared_ptr_class<accessor_type> MAcc;

  shared_ptr_class<buffer<T, buffer_dim>> MOutBufPtr;

  /// USM pointer referencing the memory to where the result of the reduction
  /// must be written. Applicable/used only for USM reductions.
  T *MUSMPointer = nullptr;

  BinaryOperation MBinaryOp;
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
/// If \c Name is undefined (is \c auto_name) then \c Type becomes the \c Name.
template <typename Name, typename Type, bool B1, bool B2, typename T3>
struct get_reduction_main_kernel_name_t {
  using name = __sycl_reduction_main_kernel<
      typename sycl::detail::get_kernel_name_t<Name, Type>::name, B1, B2, T3>;
};
template <typename Name, typename Type, bool B1, bool B2, typename T3>
struct get_reduction_aux_kernel_name_t {
  using name = __sycl_reduction_aux_kernel<
      typename sycl::detail::get_kernel_name_t<Name, Type>::name, B1, B2, T3>;
};

/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses ONEAPI::reduce() algorithm to reduce elements in each
/// of work-groups, then it calls fast SYCL atomic operations to update
/// user's reduction variable.
///
/// Briefly: calls user's lambda, ONEAPI::reduce() + atomic, INT + ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG, typename OutputT>
enable_if_t<Reduction::has_fast_reduce && Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &, OutputT Out) {
  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG, OutputT>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's function. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    typename Reduction::binary_operation BOp;
    Reducer.MValue = ONEAPI::reduce(NDIt.get_group(), Reducer.MValue, BOp);
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
          bool IsPow2WG, typename OutputT>
enable_if_t<!Reduction::has_fast_reduce && Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &, OutputT Out) {
  size_t WGSize = Range.get_local_range().size();

  // Use local memory to reduce elements in work-groups into zero-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch reduce elements that could
  // otherwise be lost in the tree-reduction algorithm used in the kernel.
  size_t NLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NLocalElements, CGH);

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG, OutputT>::name;
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

template <typename KernelName, typename KernelType, int Dims, class Reduction,
          typename OutputT>
enable_if_t<Reduction::has_fast_atomics>
reduCGFunc(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
           Reduction &Redu, OutputT Out) {

  size_t WGSize = Range.get_local_range().size();

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
/// This version uses ONEAPI::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: user's lambda, ONEAPI:reduce(), FP + ADD/MIN/MAX.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG, typename OutputT>
enable_if_t<Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &, OutputT Out) {

  size_t NWorkGroups = Range.get_group_range().size();
  // This additional check is needed for 'read_write' accessor case only.
  // It does not slow-down the kernel writing to 'discard_write' accessor as
  // the condition seems to be resolved at compile time for 'discard_write'.
  bool IsUpdateOfUserVar =
      Reduction::accessor_mode == access::mode::read_write && NWorkGroups == 1;

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG, OutputT>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    // Call user's functions. Reducer.MValue gets initialized there.
    typename Reduction::reducer_type Reducer;
    KernelFunc(NDIt, Reducer);

    // Compute the partial sum/reduction for the work-group.
    size_t WGID = NDIt.get_group_linear_id();
    typename Reduction::result_type PSum = Reducer.MValue;
    typename Reduction::binary_operation BOp;
    PSum = ONEAPI::reduce(NDIt.get_group(), PSum, BOp);
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
          bool IsPow2WG, typename OutputT>
enable_if_t<!Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &Redu, OutputT Out) {
  size_t WGSize = Range.get_local_range().size();
  size_t NWorkGroups = Range.get_group_range().size();

  bool IsUpdateOfUserVar =
      Reduction::accessor_mode == access::mode::read_write && NWorkGroups == 1;

  // Use local memory to reduce elements in work-groups into 0-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch elements that could
  // otherwise be lost in the tree-reduction algorithm.
  size_t NumLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Reduction::getReadWriteLocalAcc(NumLocalElements, CGH);
  typename Reduction::result_type ReduIdentity = Redu.getIdentity();
  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, IsPow2WG, OutputT>::name;
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

  if (Reduction::is_usm && NWorkGroups == 1) {
    if (IsPow2WG)
      reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, true>(
          CGH, KernelFunc, Range, Redu, Redu.getUSMPointer());
    else
      reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, false>(
          CGH, KernelFunc, Range, Redu, Redu.getUSMPointer());
  } else {
    auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
    if (IsPow2WG)
      reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, true>(
          CGH, KernelFunc, Range, Redu, Out);
    else
      reduCGFuncImpl<KernelName, KernelType, Dims, Reduction, false>(
          CGH, KernelFunc, Range, Redu, Out);
  }
}

/// Implements a command group function that enqueues a kernel that does one
/// iteration of reduction of elements in each of work-groups.
/// This version uses ONEAPI::reduce() algorithm to reduce elements in each
/// of work-groups. At the end of each work-groups the partial sum is written
/// to a global buffer.
///
/// Briefly: aux kernel, ONEAPI:reduce(), reproducible results, FP + ADD/MIN/MAX
template <typename KernelName, typename KernelType, bool UniformWG,
          class Reduction, typename InputT, typename OutputT>
enable_if_t<Reduction::has_fast_reduce && !Reduction::has_fast_atomics>
reduAuxCGFuncImpl(handler &CGH, size_t NWorkItems, size_t NWorkGroups,
                  size_t WGSize, Reduction &, InputT In, OutputT Out) {
  using Name = typename get_reduction_aux_kernel_name_t<
      KernelName, KernelType, Reduction::is_usm, UniformWG, OutputT>::name;
  bool IsUpdateOfUserVar =
      Reduction::accessor_mode == access::mode::read_write && NWorkGroups == 1;
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
    PSum = ONEAPI::reduce(NDIt.get_group(), PSum, BOp);
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
      Reduction::accessor_mode == access::mode::read_write && NWorkGroups == 1;

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

  // The last kernel DOES write to user's accessor passed to reduction.
  // Associate it with handler manually.
  if (NWorkGroups == 1 && !Reduction::is_usm)
    Redu.associateWithHandler(CGH);

  // The last work-group may be not fully loaded with work, or the work group
  // size may be not power of two. Those two cases considered inefficient
  // as they require additional code and checks in the kernel.
  bool HasUniformWG = NWorkGroups * WGSize == NWorkItems;
  if (!Reduction::has_fast_reduce)
    HasUniformWG = HasUniformWG && (WGSize & (WGSize - 1)) == 0;

  // Get read accessor to the buffer that was used as output
  // in the previous kernel.
  auto In = Redu.getReadAccToPreviousPartialReds(CGH);
  if (Reduction::is_usm && NWorkGroups == 1) {
    if (HasUniformWG)
      reduAuxCGFuncImpl<KernelName, KernelType, true>(
          CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Redu.getUSMPointer());
    else
      reduAuxCGFuncImpl<KernelName, KernelType, false>(
          CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Redu.getUSMPointer());
  } else {
    auto Out = Redu.getWriteAccForPartialReds(NWorkGroups, CGH);
    if (HasUniformWG)
      reduAuxCGFuncImpl<KernelName, KernelType, true>(
          CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);
    else
      reduAuxCGFuncImpl<KernelName, KernelType, false>(
          CGH, NWorkItems, NWorkGroups, WGSize, Redu, In, Out);
  }
  return NWorkGroups;
}

/// For the given 'Reductions' types pack and indices enumerating only
/// the reductions for which a local accessors are needed, this function creates
/// those local accessors and returns a tuple consisting of them.
template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::local_accessor_type...>
createReduLocalAccs(size_t Size, handler &CGH, std::index_sequence<Is...>) {
  return {Reductions::getReadWriteLocalAcc(Size, CGH)...};
}

/// For the given 'Reductions' types pack and indices enumerating them this
/// function either creates new temporary accessors for partial sums (if IsOneWG
/// is false) or returns user's accessor/USM-pointer if (IsOneWG is true).
template <bool IsOneWG, typename... Reductions, size_t... Is>
auto createReduOutAccs(size_t NWorkGroups, handler &CGH,
                       std::tuple<Reductions...> &ReduTuple,
                       std::index_sequence<Is...>) {
  return std::make_tuple(
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
  return std::make_tuple(
      std::get<Is>(ReduTuple).getReadAccToPreviousPartialReds(CGH)...);
}

template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::result_type...>
getReduIdentities(std::tuple<Reductions...> &ReduTuple,
                  std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getIdentity()...};
}

template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::binary_operation...>
getReduBOPs(std::tuple<Reductions...> &ReduTuple, std::index_sequence<Is...>) {
  return {std::get<Is>(ReduTuple).getBinaryOperation()...};
}

template <typename... Reductions, size_t... Is>
std::tuple<typename Reductions::reducer_type...>
createReducers(std::tuple<typename Reductions::result_type...> Identities,
               std::tuple<typename Reductions::binary_operation...> BOPsTuple,
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

template <bool UniformPow2WG, typename... LocalAccT, typename... ReducerT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t WGSize,
                       std::tuple<LocalAccT...> LocalAccs,
                       const std::tuple<ReducerT...> &Reducers,
                       const std::tuple<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[LID]...) =
      std::make_tuple(std::get<Is>(Reducers).MValue...);
  if (!UniformPow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <bool UniformPow2WG, typename... LocalAccT, typename... InputAccT,
          typename... ResultT, size_t... Is>
void initReduLocalAccs(size_t LID, size_t GID, size_t NWorkItems, size_t WGSize,
                       std::tuple<InputAccT...> LocalAccs,
                       std::tuple<LocalAccT...> InputAccs,
                       const std::tuple<ResultT...> Identities,
                       std::index_sequence<Is...>) {
  if (UniformPow2WG || GID < NWorkItems)
    std::tie(std::get<Is>(LocalAccs)[LID]...) =
        std::make_tuple(std::get<Is>(InputAccs)[GID]...);
  if (!UniformPow2WG)
    std::tie(std::get<Is>(LocalAccs)[WGSize]...) =
        std::make_tuple(std::get<Is>(Identities)...);
}

template <typename... LocalAccT, typename... BOPsT, size_t... Is>
void reduceReduLocalAccs(size_t IndexA, size_t IndexB,
                         std::tuple<LocalAccT...> LocalAccs,
                         std::tuple<BOPsT...> BOPs,
                         std::index_sequence<Is...>) {
  std::tie(std::get<Is>(LocalAccs)[IndexA]...) =
      std::make_tuple((std::get<Is>(BOPs)(std::get<Is>(LocalAccs)[IndexA],
                                          std::get<Is>(LocalAccs)[IndexB]))...);
}

template <bool UniformPow2WG, typename... Reductions, typename... OutAccT,
          typename... LocalAccT, typename... BOPsT, size_t... Is,
          size_t... RWIs>
void writeReduSumsToOutAccs(size_t OutAccIndex, size_t WGSize,
                            std::tuple<Reductions...> *,
                            std::tuple<OutAccT...> OutAccs,
                            std::tuple<LocalAccT...> LocalAccs,
                            std::tuple<BOPsT...> BOPs,
                            std::index_sequence<Is...>,
                            std::index_sequence<RWIs...>) {
  // This statement is needed for read_write accessors/USM-memory only.
  // It adds the initial value of the reduction variable to the final result.
  std::tie(std::get<RWIs>(LocalAccs)[0]...) =
      std::make_tuple(std::get<RWIs>(BOPs)(
          std::get<RWIs>(LocalAccs)[0],
          std::tuple_element_t<RWIs, std::tuple<Reductions...>>::getOutPointer(
              std::get<RWIs>(OutAccs))[OutAccIndex])...);

  if (UniformPow2WG) {
    std::tie(std::tuple_element_t<Is, std::tuple<Reductions...>>::getOutPointer(
        std::get<Is>(OutAccs))[OutAccIndex]...) =
        std::make_tuple(std::get<Is>(LocalAccs)[0]...);
  } else {
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

struct IsRWReductionPredicate {
  template <typename T> struct Func {
    static constexpr bool value =
        std::remove_pointer_t<T>::accessor_mode == access::mode::read_write;
  };
};

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

template <typename KernelName, bool UniformPow2WG, bool IsOneWG,
          typename KernelType, int Dims, typename... Reductions, size_t... Is>
void reduCGFuncImpl(handler &CGH, KernelType KernelFunc,
                    const nd_range<Dims> &Range,
                    std::tuple<Reductions...> &ReduTuple,
                    std::index_sequence<Is...> ReduIndices) {

  size_t WGSize = Range.get_local_range().size();
  size_t LocalAccSize = WGSize + (UniformPow2WG ? 0 : 1);
  auto LocalAccsTuple =
      createReduLocalAccs<Reductions...>(LocalAccSize, CGH, ReduIndices);

  size_t NWorkGroups = IsOneWG ? 1 : Range.get_group_range().size();
  auto OutAccsTuple =
      createReduOutAccs<IsOneWG>(NWorkGroups, CGH, ReduTuple, ReduIndices);
  auto IdentitiesTuple = getReduIdentities(ReduTuple, ReduIndices);
  auto BOPsTuple = getReduBOPs(ReduTuple, ReduIndices);

  using Name = typename get_reduction_main_kernel_name_t<
      KernelName, KernelType, UniformPow2WG, IsOneWG,
      std::tuple<Reductions...>>::name;
  CGH.parallel_for<Name>(Range, [=](nd_item<Dims> NDIt) {
    auto ReduIndices = std::index_sequence_for<Reductions...>();
    auto ReducersTuple =
        createReducers<Reductions...>(IdentitiesTuple, BOPsTuple, ReduIndices);
    // The .MValue field of each of the elements in ReducersTuple
    // gets initialized in this call.
    callReduUserKernelFunc(KernelFunc, NDIt, ReducersTuple, ReduIndices);

    size_t WGSize = NDIt.get_local_range().size();
    size_t LID = NDIt.get_local_linear_id();
    initReduLocalAccs<UniformPow2WG>(LID, WGSize, LocalAccsTuple, ReducersTuple,
                                     IdentitiesTuple, ReduIndices);
    NDIt.barrier();

    size_t PrevStep = WGSize;
    for (size_t CurStep = PrevStep >> 1; CurStep > 0; CurStep >>= 1) {
      if (LID < CurStep) {
        // LocalReds[LID] = BOp(LocalReds[LID], LocalReds[LID + CurStep]);
        reduceReduLocalAccs(LID, LID + CurStep, LocalAccsTuple, BOPsTuple,
                            ReduIndices);
      } else if (!UniformPow2WG && LID == CurStep && (PrevStep & 0x1)) {
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
      // If there is only one work-group, then the original accessors need to be
      // updated, i.e. after the work in each work-group is done, the work-group
      // result is added to the original value of the read-write accessors or
      // USM memory.
      std::conditional_t<IsOneWG, IsRWReductionPredicate,
                         EmptyReductionPredicate>
          Predicate;
      auto RWReduIndices =
          filterSequence<Reductions...>(Predicate, ReduIndices);
      writeReduSumsToOutAccs<UniformPow2WG>(
          GrID, WGSize, (std::tuple<Reductions...> *)nullptr, OutAccsTuple,
          LocalAccsTuple, BOPsTuple, ReduIndices, RWReduIndices);
    }
  });
}

template <typename KernelName, typename KernelType, int Dims,
          typename... Reductions, size_t... Is>
void reduCGFunc(handler &CGH, KernelType KernelFunc,
                const nd_range<Dims> &Range,
                std::tuple<Reductions...> &ReduTuple,
                std::index_sequence<Is...> ReduIndices) {
  size_t NWorkItems = Range.get_global_range().size();
  size_t WGSize = Range.get_local_range().size();
  size_t NWorkGroups = Range.get_group_range().size();

  bool Pow2WG = (WGSize & (WGSize - 1)) == 0;
  bool HasUniformWG = Pow2WG && (NWorkGroups * WGSize == NWorkItems);
  if (NWorkGroups == 1) {
    if (HasUniformWG)
      reduCGFuncImpl<KernelName, true, true>(CGH, KernelFunc, Range, ReduTuple,
                                             ReduIndices);
    else
      reduCGFuncImpl<KernelName, false, true>(CGH, KernelFunc, Range, ReduTuple,
                                              ReduIndices);
  } else {
    if (HasUniformWG)
      reduCGFuncImpl<KernelName, true, false>(CGH, KernelFunc, Range, ReduTuple,
                                              ReduIndices);
    else
      reduCGFuncImpl<KernelName, false, false>(CGH, KernelFunc, Range,
                                               ReduTuple, ReduIndices);
  }
}

// The list of reductions may be empty; for such cases there is nothing to do.
// This function is intentionally made template to eliminate the need in holding
// it in SYCL library, what would be less efficient and also would create the
// need in keeping it for long due support backward ABI compatibility.
template <typename HandlerT>
std::enable_if_t<std::is_same<HandlerT, handler>::value>
associateReduAccsWithHandlerHelper(HandlerT &) {}

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

  using Name =
      typename get_reduction_aux_kernel_name_t<KernelName, KernelType,
                                               UniformPow2WG, IsOneWG,
                                               std::tuple<Reductions...>>::name;
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
      // If there is only one work-group, then the original accessors need to be
      // updated, i.e. after the work in each work-group is done, the work-group
      // result is added to the original value of the read-write accessors or
      // USM memory.
      std::conditional_t<IsOneWG, IsRWReductionPredicate,
                         EmptyReductionPredicate>
          Predicate;
      auto RWReduIndices =
          filterSequence<Reductions...>(Predicate, ReduIndices);
      writeReduSumsToOutAccs<UniformPow2WG>(
          GrID, WGSize, (std::tuple<Reductions...> *)nullptr, OutAccsTuple,
          LocalAccsTuple, BOPsTuple, ReduIndices, RWReduIndices);
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
detail::reduction_impl<T, BinaryOperation, Dims, false, AccMode, IsPH>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          const T &Identity, BinaryOperation BOp) {
  // The Combiner argument was needed only to define the BinaryOperation param.
  return detail::reduction_impl<T, BinaryOperation, Dims, false, AccMode, IsPH>(
      Acc, Identity, BOp);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc and the binary operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH>
detail::enable_if_t<
    detail::IsKnownIdentityOp<T, BinaryOperation>::value,
    detail::reduction_impl<T, BinaryOperation, Dims, false, AccMode, IsPH>>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          BinaryOperation) {
  // The Combiner argument was needed only to define the BinaryOperation param.
  return detail::reduction_impl<T, BinaryOperation, Dims, false, AccMode, IsPH>(
      Acc);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 3 arguments: the reference to the reduction variable to where
/// the computed reduction must be stored \param VarPtr, identity value
/// \param Identity, and the binary operation used in the reduction.
template <typename T, class BinaryOperation>
detail::reduction_impl<T, BinaryOperation, 0, true, access::mode::read_write>
reduction(T *VarPtr, const T &Identity, BinaryOperation BOp) {
  return detail::reduction_impl<T, BinaryOperation, 0, true,
                                access::mode::read_write>(VarPtr, Identity,
                                                          BOp);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the reference to the reduction variable, to where
/// the computed reduction must be stored \param VarPtr, and the binary
/// operation used in the reduction.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation>
detail::enable_if_t<detail::IsKnownIdentityOp<T, BinaryOperation>::value,
                    detail::reduction_impl<T, BinaryOperation, 0, true,
                                           access::mode::read_write>>
reduction(T *VarPtr, BinaryOperation) {
  return detail::reduction_impl<T, BinaryOperation, 0, true,
                                access::mode::read_write>(VarPtr);
}

} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
