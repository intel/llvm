//==----- atomic_ref.hpp - SYCL 2020 atomic_ref ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/memory_enums.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/multi_ptr.hpp>
#endif

#ifndef __SYCL_DEVICE_ONLY__
#include <atomic>
#endif
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using memory_order = cl::sycl::memory_order;
using memory_scope = cl::sycl::memory_scope;

template <typename T> struct IsValidAtomicRefType {
  static constexpr bool value =
      (std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
       std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
       std::is_same<T, long long>::value ||
       std::is_same<T, unsigned long long>::value ||
       std::is_same<T, float>::value || std::is_same<T, double>::value ||
       std::is_pointer<T>::value);
};

template <cl::sycl::access::address_space AS>
struct IsValidAtomicRefAddressSpace {
  static constexpr bool value =
      (AS == access::address_space::global_space ||
       AS == access::address_space::local_space ||
       AS == access::address_space::global_device_space ||
       AS == access::address_space::generic_space);
};

// DefaultOrder parameter is limited to read-modify-write orders
template <memory_order Order>
using IsValidDefaultOrder = bool_constant<Order == memory_order::relaxed ||
                                          Order == memory_order::acq_rel ||
                                          Order == memory_order::seq_cst>;

template <memory_order ReadModifyWriteOrder> struct memory_order_traits;

template <> struct memory_order_traits<memory_order::relaxed> {
  static constexpr memory_order read_order = memory_order::relaxed;
  static constexpr memory_order write_order = memory_order::relaxed;
};

template <> struct memory_order_traits<memory_order::acq_rel> {
  static constexpr memory_order read_order = memory_order::acquire;
  static constexpr memory_order write_order = memory_order::release;
};

template <> struct memory_order_traits<memory_order::seq_cst> {
  static constexpr memory_order read_order = memory_order::seq_cst;
  static constexpr memory_order write_order = memory_order::seq_cst;
};

inline constexpr memory_order getLoadOrder(memory_order order) {
  switch (order) {
  case memory_order::relaxed:
    return memory_order::relaxed;

  case memory_order::acquire:
  case memory_order::__consume_unsupported:
  case memory_order::acq_rel:
  case memory_order::release:
    return memory_order::acquire;

  case memory_order::seq_cst:
    return memory_order::seq_cst;
  }
}

template <typename T, typename = void> struct bit_equal;

template <typename T>
struct bit_equal<T, typename detail::enable_if_t<std::is_integral<T>::value>> {
  bool operator()(const T &lhs, const T &rhs) { return lhs == rhs; }
};

template <> struct bit_equal<float> {
  bool operator()(const float &lhs, const float &rhs) {
    auto LhsInt = sycl::bit_cast<uint32_t>(lhs);
    auto RhsInt = sycl::bit_cast<uint32_t>(rhs);
    return LhsInt == RhsInt;
  }
};

template <> struct bit_equal<double> {
  bool operator()(const double &lhs, const double &rhs) {
    auto LhsInt = sycl::bit_cast<uint64_t>(lhs);
    auto RhsInt = sycl::bit_cast<uint64_t>(rhs);
    return LhsInt == RhsInt;
  }
};

// Functionality for any atomic of type T, reused by partial specializations
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref_base {
  static_assert(
      detail::IsValidAtomicRefType<T>::value,
      "Invalid atomic type.  Valid types are int, unsigned int, long, "
      "unsigned long, long long, unsigned long long, float, double "
      "and pointer types");
  static_assert(
      detail::IsValidAtomicRefAddressSpace<AddressSpace>::value,
      "Invalid atomic address_space.  Valid address spaces are: "
      "global_space, local_space, global_device_space, generic_space");
  static_assert(
      detail::IsValidDefaultOrder<DefaultOrder>::value,
      "Invalid default memory_order for atomics.  Valid defaults are: "
      "relaxed, acq_rel, seq_cst");

public:
  using value_type = T;
  static constexpr size_t required_alignment = sizeof(T);
  static constexpr bool is_always_lock_free =
      detail::IsValidAtomicRefType<T>::value;
  static constexpr memory_order default_read_order =
      detail::memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      detail::memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept {
    return detail::IsValidAtomicRefType<T>::value;
  }

#ifdef __SYCL_DEVICE_ONLY__
  explicit atomic_ref_base(T &ref) : ptr(multi_ptr<T, AddressSpace>(&ref)) {}
#else
  // FIXME: This reinterpret_cast is UB, but happens to work for now
  explicit atomic_ref_base(T &ref)
      : ptr(reinterpret_cast<std::atomic<T> *>(&ref)) {}
#endif
  // Our implementation of copy constructor could be trivial
  // Defined this way for consistency with standard atomic_ref
  atomic_ref_base(const atomic_ref_base &ref) noexcept { ptr = ref.ptr; };
  atomic_ref_base &operator=(const atomic_ref_base &) = delete;

  void store(T operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    detail::spirv::AtomicStore(ptr, scope, order, operand);
#else
    (void)scope;
    ptr->store(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator=(T desired) const noexcept {
    store(desired);
    return desired;
  }

  T load(memory_order order = default_read_order,
         memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicLoad(ptr, scope, order);
#else
    (void)scope;
    return ptr->load(detail::getStdMemoryOrder(order));
#endif
  }

  operator T() const noexcept { return load(); }

  T exchange(T operand, memory_order order = default_read_modify_write_order,
             memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicExchange(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->exchange(operand, detail::getStdMemoryOrder(order));
#endif
  }

  bool
  compare_exchange_strong(T &expected, T desired, memory_order success,
                          memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    T value = detail::spirv::AtomicCompareExchange(ptr, scope, success, failure,
                                                   desired, expected);
    bool succeeded = detail::bit_equal<T>()(value, expected);
    if (!succeeded) {
      expected = value;
    }
    return succeeded;
#else
    (void)scope;
    return ptr->compare_exchange_strong(expected, desired,
                                        detail::getStdMemoryOrder(success),
                                        detail::getStdMemoryOrder(failure));
#endif
  }

  bool
  compare_exchange_strong(T &expected, T desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  bool
  compare_exchange_weak(T &expected, T desired, memory_order success,
                        memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    // SPIR-V AtomicCompareExchangeWeak is deprecated and equivalent to
    // AtomicCompareExchange. For now, use AtomicCompareExchange on device and
    // compare_exchange_weak on host
#ifdef __SYCL_DEVICE_ONLY__
    return compare_exchange_strong(expected, desired, success, failure, scope);
#else
    (void)scope;
    return ptr->compare_exchange_weak(expected, desired,
                                      detail::getStdMemoryOrder(success),
                                      detail::getStdMemoryOrder(failure));
#endif
  }

  bool
  compare_exchange_weak(T &expected, T desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

protected:
#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr<T, AddressSpace> ptr;
#else
  std::atomic<T> *ptr;
#endif
};

// Hook allowing partial specializations to inherit atomic_ref_base
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace, typename = void>
class atomic_ref_impl
    : public atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace> {
public:
  using atomic_ref_base<T, DefaultOrder, DefaultScope,
                        AddressSpace>::atomic_ref_base;
};

// Partial specialization for integral types
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref_impl<T, DefaultOrder, DefaultScope, AddressSpace,
                      typename detail::enable_if_t<std::is_integral<T>::value>>
    : public atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace> {

public:
  using value_type = T;
  using difference_type = value_type;
  static constexpr size_t required_alignment = sizeof(T);
  static constexpr bool is_always_lock_free =
      detail::IsValidAtomicRefType<T>::value;
  static constexpr memory_order default_read_order =
      detail::memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      detail::memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  using atomic_ref_base<T, DefaultOrder, DefaultScope,
                        AddressSpace>::atomic_ref_base;
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::load;
  using atomic_ref_base<T, DefaultOrder, DefaultScope,
                        AddressSpace>::compare_exchange_weak;
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::operator=;

  T fetch_add(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicIAdd(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->fetch_add(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator+=(T operand) const noexcept {
    return fetch_add(operand) + operand;
  }

  T operator++(int) const noexcept {
    // TODO: use AtomicIIncrement as an optimization
    return fetch_add(1);
  }

  T operator++() const noexcept {
    // TODO: use AtomicIIncrement as an optimization
    return fetch_add(1) + 1;
  }

  T fetch_sub(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicISub(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->fetch_sub(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator-=(T operand) const noexcept {
    return fetch_sub(operand) - operand;
  }

  T operator--(int) const noexcept {
    // TODO: use AtomicIDecrement as an optimization
    return fetch_sub(1);
  }

  T operator--() const noexcept {
    // TODO: use AtomicIDecrement as an optimization
    return fetch_sub(1) - 1;
  }

  T fetch_and(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicAnd(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->fetch_and(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator&=(T operand) const noexcept {
    return fetch_and(operand) & operand;
  }

  T fetch_or(T operand, memory_order order = default_read_modify_write_order,
             memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicOr(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->fetch_or(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator|=(T operand) const noexcept { return fetch_or(operand) | operand; }

  T fetch_xor(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicXor(ptr, scope, order, operand);
#else
    (void)scope;
    return ptr->fetch_xor(operand, detail::getStdMemoryOrder(order));
#endif
  }

  T operator^=(T operand) const noexcept {
    return fetch_xor(operand) ^ operand;
  }

  T fetch_min(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicMin(ptr, scope, order, operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T old = load(load_order, scope);
    while (operand < old &&
           !compare_exchange_weak(old, operand, order, scope)) {
    }
    return old;
#endif
  }

  T fetch_max(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::spirv::AtomicMax(ptr, scope, order, operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T old = load(load_order, scope);
    while (operand > old &&
           !compare_exchange_weak(old, operand, order, scope)) {
    }
    return old;
#endif
  }

private:
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::ptr;
};

// Partial specialization for floating-point types
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref_impl<
    T, DefaultOrder, DefaultScope, AddressSpace,
    typename detail::enable_if_t<std::is_floating_point<T>::value>>
    : public atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace> {

public:
  using value_type = T;
  using difference_type = value_type;
  static constexpr size_t required_alignment = sizeof(T);
  static constexpr bool is_always_lock_free =
      detail::IsValidAtomicRefType<T>::value;
  static constexpr memory_order default_read_order =
      detail::memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      detail::memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  using atomic_ref_base<T, DefaultOrder, DefaultScope,
                        AddressSpace>::atomic_ref_base;
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::load;
  using atomic_ref_base<T, DefaultOrder, DefaultScope,
                        AddressSpace>::compare_exchange_weak;
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::operator=;

  T fetch_add(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
// TODO: Remove the "native atomics" macro check once implemented for all
// backends
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_USE_NATIVE_FP_ATOMICS)
    return detail::spirv::AtomicFAdd(ptr, scope, order, operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T expected;
    T desired;
    do {
      expected =
          load(load_order, scope); // performs better with load in CAS loop.
      desired = expected + operand;
    } while (!compare_exchange_weak(expected, desired, order, scope));
    return expected;
#endif
  }

  T operator+=(T operand) const noexcept {
    return fetch_add(operand) + operand;
  }

  T fetch_sub(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
// TODO: Remove the "native atomics" macro check once implemented for all
// backends
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_USE_NATIVE_FP_ATOMICS)
    return detail::spirv::AtomicFAdd(ptr, scope, order, -operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T expected = load(load_order, scope);
    T desired;
    do {
      desired = expected - operand;
    } while (!compare_exchange_weak(expected, desired, order, scope));
    return expected;
#endif
  }

  T operator-=(T operand) const noexcept {
    return fetch_sub(operand) - operand;
  }

  T fetch_min(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
// TODO: Remove the "native atomics" macro check once implemented for all
// backends
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_USE_NATIVE_FP_ATOMICS)
    return detail::spirv::AtomicMin(ptr, scope, order, operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T old = load(load_order, scope);
    while (operand < old &&
           !compare_exchange_weak(old, operand, order, scope)) {
    }
    return old;
#endif
  }

  T fetch_max(T operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
// TODO: Remove the "native atomics" macro check once implemented for all
// backends
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_USE_NATIVE_FP_ATOMICS)
    return detail::spirv::AtomicMax(ptr, scope, order, operand);
#else
    auto load_order = detail::getLoadOrder(order);
    T old = load(load_order, scope);
    while (operand > old &&
           !compare_exchange_weak(old, operand, order, scope)) {
    }
    return old;
#endif
  }

private:
  using atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>::ptr;
};

// Partial specialization for pointer types
// Arithmetic is emulated because target's representation of T* is unknown
// TODO: Find a way to use intptr_t or uintptr_t atomics instead
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref_impl<T *, DefaultOrder, DefaultScope, AddressSpace>
    : public atomic_ref_base<uintptr_t, DefaultOrder, DefaultScope,
                             AddressSpace> {

private:
  using base_type =
      atomic_ref_base<uintptr_t, DefaultOrder, DefaultScope, AddressSpace>;

public:
  using value_type = T *;
  using difference_type = ptrdiff_t;
  static constexpr size_t required_alignment = sizeof(T *);
  static constexpr bool is_always_lock_free =
      detail::IsValidAtomicRefType<T>::value;
  static constexpr memory_order default_read_order =
      detail::memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      detail::memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  using base_type::is_lock_free;

  atomic_ref_impl(T *&ref) : base_type(reinterpret_cast<uintptr_t &>(ref)) {}

  void store(T *operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    base_type::store(reinterpret_cast<uintptr_t>(operand), order, scope);
  }

  T *operator=(T *desired) const noexcept {
    store(desired);
    return desired;
  }

  T *load(memory_order order = default_read_order,
          memory_scope scope = default_scope) const noexcept {
    return reinterpret_cast<T *>(base_type::load(order, scope));
  }

  operator T *() const noexcept { return load(); }

  T *exchange(T *operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
    return reinterpret_cast<T *>(base_type::exchange(
        reinterpret_cast<uintptr_t>(operand), order, scope));
  }

  T *fetch_add(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    // TODO: Find a way to avoid compare_exchange here
    auto load_order = detail::getLoadOrder(order);
    T *expected;
    T *desired;
    do {
      expected = load(load_order, scope);
      desired = expected + operand;
    } while (!compare_exchange_weak(expected, desired, order, scope));
    return expected;
  }

  T *operator+=(difference_type operand) const noexcept {
    return fetch_add(operand) + operand;
  }

  T *operator++(int) const noexcept { return fetch_add(difference_type(1)); }

  T *operator++() const noexcept {
    return fetch_add(difference_type(1)) + difference_type(1);
  }

  T *fetch_sub(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    // TODO: Find a way to avoid compare_exchange here
    auto load_order = detail::getLoadOrder(order);
    T *expected = load(load_order, scope);
    T *desired;
    do {
      desired = expected - operand;
    } while (!compare_exchange_weak(expected, desired, order, scope));
    return expected;
  }

  T *operator-=(difference_type operand) const noexcept {
    return fetch_sub(operand) - operand;
  }

  T *operator--(int) const noexcept { return fetch_sub(difference_type(1)); }

  T *operator--() const noexcept {
    return fetch_sub(difference_type(1)) - difference_type(1);
  }

  bool
  compare_exchange_strong(T *&expected, T *desired, memory_order success,
                          memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
    return base_type::compare_exchange_strong(
        reinterpret_cast<uintptr_t &>(expected),
        reinterpret_cast<uintptr_t>(desired), success, failure, scope);
  }

  bool
  compare_exchange_strong(T *&expected, T *desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  bool
  compare_exchange_weak(T *&expected, T *desired, memory_order success,
                        memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    return base_type::compare_exchange_weak(
        reinterpret_cast<uintptr_t &>(expected),
        reinterpret_cast<uintptr_t>(desired), success, failure, scope);
  }

  bool
  compare_exchange_weak(T *&expected, T *desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

private:
  using base_type::ptr;
};

} // namespace detail

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace =
              access::address_space::generic_space>
class atomic_ref : public detail::atomic_ref_impl<T, DefaultOrder, DefaultScope,
                                                  AddressSpace> {
public:
  using detail::atomic_ref_impl<T, DefaultOrder, DefaultScope,
                                AddressSpace>::atomic_ref_impl;
  using detail::atomic_ref_impl<T, DefaultOrder, DefaultScope,
                                AddressSpace>::operator=;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
