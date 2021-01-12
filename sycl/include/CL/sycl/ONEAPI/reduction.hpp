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

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity_impl
    : std::integral_constant<
          bool, IsKnownIdentityOp<AccumulatorT, BinaryOperation>::value> {};

template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct known_identity_impl {};

/// Returns zero as identity for ADD, OR, XOR operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsZeroIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = 0;
};

template <typename BinaryOperation>
struct known_identity_impl<BinaryOperation, half,
                           typename std::enable_if<IsZeroIdentityOp<
                               half, BinaryOperation>::value>::type> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      0;
#else
      cl::sycl::detail::host_half_impl::half(static_cast<uint16_t>(0));
#endif
};

/// Returns one as identify for MULTIPLY operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsOneIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = 1;
};

template <typename BinaryOperation>
struct known_identity_impl<BinaryOperation, half,
                           typename std::enable_if<IsOneIdentityOp<
                               half, BinaryOperation>::value>::type> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      1;
#else
      cl::sycl::detail::host_half_impl::half(static_cast<uint16_t>(0x3C00));
#endif
};

/// Returns bit image consisting of all ones as identity for AND operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsOnesIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = ~static_cast<AccumulatorT>(0);
};

/// Returns maximal possible value as identity for MIN operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsMinimumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value =
      std::numeric_limits<AccumulatorT>::has_infinity
          ? std::numeric_limits<AccumulatorT>::infinity()
          : (std::numeric_limits<AccumulatorT>::max)();
};

/// Returns minimal possible value as identity for MAX operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsMaximumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value =
      std::numeric_limits<AccumulatorT>::has_infinity
          ? static_cast<AccumulatorT>(
                -std::numeric_limits<AccumulatorT>::infinity())
          : std::numeric_limits<AccumulatorT>::lowest();
};

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

  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<has_known_identity_impl<_BinaryOperation, _T>::value, _T>
  getIdentity() {
    return known_identity_impl<_BinaryOperation, _T>::value;
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

/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, bool IsUSM,
          access::mode AccMode = access::mode::read_write,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class reduction_impl {
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

  accessor<T, buffer_dim, access::mode::discard_read_write,
           access::target::local>
  getReadWriteLocalAcc(size_t Size, handler &CGH) {
    return accessor<T, buffer_dim, access::mode::discard_read_write,
                    access::target::local>(Size, CGH);
  }

  accessor<T, buffer_dim, access::mode::read>
  getReadAccToPreviousPartialReds(handler &CGH) const {
    CGH.addReduction(MOutBufPtr);
    return accessor<T, buffer_dim, access::mode::read>(*MOutBufPtr, CGH);
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
/// \c Name and \c Type types: if \c Name is undefined (is a \c auto_name) then
/// \c Type becomes the \c Name.
template <typename Name, typename Type, bool B1, bool B2, typename OutputT>
struct get_reduction_main_kernel_name_t {
  using name = __sycl_reduction_main_kernel<
      typename sycl::detail::get_kernel_name_t<Name, Type>::name, B1, B2,
      OutputT>;
};
template <typename Name, typename Type, bool B1, bool B2, typename OutputT>
struct get_reduction_aux_kernel_name_t {
  using name = __sycl_reduction_aux_kernel<
      typename sycl::detail::get_kernel_name_t<Name, Type>::name, B1, B2,
      OutputT>;
};

/// Implements a command group function that enqueues a kernel that calls
/// user's lambda function KernelFunc and also does one iteration of reduction
/// of elements computed in user's lambda function.
/// This version uses ONEAPI::reduce() algorithm to reduce elements in each
/// of work-groups, then it calls fast sycl atomic operations to update
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
/// of work-groups, then it calls fast sycl atomic operations to update
/// user's reduction variable.
///
/// Briefly: calls user's lambda, tree-reduction + atomic, INT + AND/OR/XOR.
template <typename KernelName, typename KernelType, int Dims, class Reduction,
          bool IsPow2WG, typename OutputT>
enable_if_t<!Reduction::has_fast_reduce && Reduction::has_fast_atomics>
reduCGFuncImpl(handler &CGH, KernelType KernelFunc, const nd_range<Dims> &Range,
               Reduction &Redu, OutputT Out) {
  size_t WGSize = Range.get_local_range().size();

  // Use local memory to reduce elements in work-groups into zero-th element.
  // If WGSize is not power of two, then WGSize+1 elements are allocated.
  // The additional last element is used to catch reduce elements that could
  // otherwise be lost in the tree-reduction algorithm used in the kernel.
  size_t NLocalElements = WGSize + (IsPow2WG ? 0 : 1);
  auto LocalReds = Redu.getReadWriteLocalAcc(NLocalElements, CGH);

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
  auto LocalReds = Redu.getReadWriteLocalAcc(NumLocalElements, CGH);
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
  auto LocalReds = Redu.getReadWriteLocalAcc(NumLocalElements, CGH);

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

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity : detail::has_known_identity_impl<
                                typename std::decay<BinaryOperation>::type,
                                typename std::decay<AccumulatorT>::type> {};
#if __cplusplus >= 201703L
template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;
#endif

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity
    : detail::known_identity_impl<typename std::decay<BinaryOperation>::type,
                                  typename std::decay<AccumulatorT>::type> {};
#if __cplusplus >= 201703L
template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;
#endif

} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
