//==---------------- reduction.hpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/accessor.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

namespace detail {

using cl::sycl::detail::bool_constant;
using cl::sycl::detail::enable_if_t;
using cl::sycl::detail::is_geninteger16bit;
using cl::sycl::detail::is_geninteger32bit;
using cl::sycl::detail::is_geninteger64bit;
using cl::sycl::detail::is_geninteger8bit;
using cl::sycl::detail::remove_AS;

// Identity = 0
template <typename T, class BinaryOperation>
using IsZeroIdentityOp = bool_constant<
    ((is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
      is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
     (std::is_same<BinaryOperation, intel::plus<T>>::value ||
      std::is_same<BinaryOperation, intel::bit_or<T>>::value ||
      std::is_same<BinaryOperation, intel::bit_xor<T>>::value)) ||
    ((std::is_same<T, float>::value || std::is_same<T, double>::value) &&
     std::is_same<BinaryOperation, intel::plus<T>>::value)>;

// Identity = 1
template <typename T, class BinaryOperation>
using IsOneIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, float>::value || std::is_same<T, double>::value) &&
    std::is_same<BinaryOperation, std::multiplies<T>>::value>;

// Identity = ~0
template <typename T, class BinaryOperation>
using IsOnesIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
    std::is_same<BinaryOperation, intel::bit_and<T>>::value>;

// Identity = <max possible value>
template <typename T, class BinaryOperation>
using IsMinimumIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, float>::value || std::is_same<T, double>::value) &&
    std::is_same<BinaryOperation, intel::minimum<T>>::value>;

// Identity = <min possible value>
template <typename T, class BinaryOperation>
using IsMaximumIdentityOp = bool_constant<
    (is_geninteger8bit<T>::value || is_geninteger16bit<T>::value ||
     is_geninteger32bit<T>::value || is_geninteger64bit<T>::value ||
     std::is_same<T, float>::value || std::is_same<T, double>::value) &&
    std::is_same<BinaryOperation, intel::maximum<T>>::value>;

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
  reducer(const T &Identity) : MValue(Identity), MIdentity(Identity) {}
  void combine(const T &Partial) {
    BinaryOperation BOp;
    MValue = BOp(MValue, Partial);
  }

  T getIdentity() const { return MIdentity; }

  T MValue;

private:
  const T MIdentity;
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
/// For example, it is known that 0 is identity for intel::plus operations
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
  reducer(const T &Identity) : MValue(getIdentity()) {}

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
    return (std::numeric_limits<_T>::max)();
  }

  /// Returns minimal possible value as identity for MAX operations.
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  static enable_if_t<IsMaximumIdentityOp<_T, _BinaryOperation>::value, _T>
  getIdentity() {
    return (std::numeric_limits<_T>::min)();
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  std::is_same<BinaryOperation, intel::plus<T>>::value,
              reducer &>
  operator+=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  std::is_same<BinaryOperation, std::multiplies<T>>::value,
              reducer &>
  operator*=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  std::is_same<BinaryOperation, intel::bit_or<T>>::value,
              reducer &>
  operator|=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  std::is_same<BinaryOperation, intel::bit_xor<T>>::value,
              reducer &>
  operator^=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  template <typename _T = T>
  enable_if_t<std::is_same<_T, T>::value &&
                  std::is_same<BinaryOperation, intel::bit_and<T>>::value,
              reducer &>
  operator&=(const _T &Partial) {
    combine(Partial);
    return *this;
  }

  /// Atomic ADD operation: *ReduVarPtr += MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::plus<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_add(MValue);
  }

  /// Atomic BITWISE OR operation: *ReduVarPtr |= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::bit_or<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_or(MValue);
  }

  /// Atomic BITWISE XOR operation: *ReduVarPtr ^= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::bit_xor<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_xor(MValue);
  }

  /// Atomic BITWISE AND operation: *ReduVarPtr &= MValue;
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::bit_and<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_and(MValue);
  }

  /// Atomic MIN operation: *ReduVarPtr = intel::minimum(*ReduVarPtr, MValue);
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::minimum<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_min(MValue);
  }

  /// Atomic MAX operation: *ReduVarPtr = intel::maximum(*ReduVarPtr, MValue);
  template <typename _T = T, class _BinaryOperation = BinaryOperation>
  enable_if_t<std::is_same<typename remove_AS<_T>::type, T>::value &&
              (is_geninteger32bit<T>::value || is_geninteger64bit<T>::value) &&
              std::is_same<_BinaryOperation, intel::maximum<T>>::value>
  atomic_combine(_T *ReduVarPtr) const {
    atomic<T, access::address_space::global_space>(global_ptr<T>(ReduVarPtr))
        .fetch_max(MValue);
  }

  T MValue;
};

/// This class encapsulates the reduction variable/accessor,
/// the reduction operator and an optional operator identity.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPlaceholder>
class reduction_impl {
public:
  using reducer_type = reducer<T, BinaryOperation>;
  using result_type = T;
  using binary_operation = BinaryOperation;
  using accessor_type =
      accessor<T, Dims, AccMode, access::target::global_buffer, IsPlaceholder>;
  static constexpr access::mode accessor_mode = AccMode;
  static constexpr int accessor_dim = Dims;
  static constexpr int buffer_dim = (Dims == 0) ? 1 : Dims;

  // Only scalar (i.e. 0-dim and 1-dim with 1 element) reductions supported now.
  // TODO: suport (Dims > 1) and placeholder accessors/reductions.
  // TODO: support true 1-Dimensional accessors/reductions (get_count() > 1).
  // (get_count() == 1) is checked in the constructor of reduction_impl.
  static_assert(Dims <= 1 && IsPlaceholder == access::placeholder::false_t,
                "Multi-dimensional and placeholder reductions"
                " are not supported yet.");

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
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc) : MAcc(Acc), MIdentity(getIdentity()) {
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
  }

  /// Constructs reduction_impl when the identity value is statically known,
  /// and user still passed the identity value.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc, const T &Identity)
      : MAcc(Acc), MIdentity(Identity) {
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
    // For operations with known identity value the operator == is defined.
    // It is sort of dilemma here: from one point of view - user may set
    // such identity that would be enough for his data, i.e. identity=100 for
    // min operation if user knows all data elements are less than 100.
    // From another point of view - it is the source of unexpected errors,
    // when the input data changes.
    // Let's be strict for now and emit an error if identity is not proper.
    assert(Identity == getIdentity() && "Unexpected Identity parameter value.");
  }

  /// Constructs reduction_impl when the identity value is unknown.
  template <
      typename _T = T, class _BinaryOperation = BinaryOperation,
      enable_if_t<!IsKnownIdentityOp<_T, _BinaryOperation>::value> * = nullptr>
  reduction_impl(accessor_type &Acc, const T &Identity)
      : MAcc(Acc), MIdentity(Identity) {
    assert(Acc.get_count() == 1 &&
           "Only scalar/1-element reductions are supported now.");
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

  accessor_type getWriteAccForPartialReds(size_t Size, size_t RunNumber,
                                          handler &CGH) {
    if (Size == 1) {
      if (RunNumber > 0)
        CGH.associateWithHandler(this->MAcc);
      return this->MAcc;
    }
    // Create a new output buffer and return an accessor to it.
    MOutBufPtr = std::make_shared<buffer<T, buffer_dim>>(range<1>(Size));
    CGH.addReduction(MOutBufPtr);
    return accessor_type(*MOutBufPtr, CGH);
  }
  /// User's accessor to where the reduction must be written.
  accessor_type MAcc;

private:
  /// Identity of the BinaryOperation.
  /// The result of BinaryOperation(X, MIdentity) is equal to X for any X.
  const T MIdentity;
  shared_ptr_class<buffer<T, buffer_dim>> MOutBufPtr;
};

} // namespace detail

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 3 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc, identity value \param Identity, and the
/// binary operation that must be used in the reduction \param Combiner.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH>
detail::reduction_impl<T, BinaryOperation, Dims, AccMode, IsPH>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          const T &Identity, BinaryOperation Combiner) {
  // The Combiner argument was needed only to define the BinaryOperation param.
  return detail::reduction_impl<T, BinaryOperation, Dims, AccMode, IsPH>(
      Acc, Identity);
}

/// Creates and returns an object implementing the reduction functionality.
/// Accepts 2 arguments: the accessor to buffer to where the computed reduction
/// must be stored \param Acc and the binary operation that must be used
/// in the reduction \param Combiner.
/// The identity value is not passed to this version as it is statically known.
template <typename T, class BinaryOperation, int Dims, access::mode AccMode,
          access::placeholder IsPH>
detail::enable_if_t<
    detail::IsKnownIdentityOp<T, BinaryOperation>::value,
    detail::reduction_impl<T, BinaryOperation, Dims, AccMode, IsPH>>
reduction(accessor<T, Dims, AccMode, access::target::global_buffer, IsPH> &Acc,
          BinaryOperation Combiner) {
  // The Combiner argument was needed only to define the BinaryOperation param.
  return detail::reduction_impl<T, BinaryOperation, Dims, AccMode, IsPH>(Acc);
}

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
