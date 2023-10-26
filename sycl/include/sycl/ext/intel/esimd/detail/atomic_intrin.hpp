//==-------- atomic_intrin.hpp - Atomic intrinsic definition file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

/// @cond ESIMD_DETAIL

namespace __ESIMD_DNS {

// This function implements atomic update of pre-existing variable in the
// absense of C++ 20's atomic_ref.

// __atomic_* functions support only integral types. In order to
// support floating types for certain operations like min/max,
// 'cmpxchg' operation is applied for result values using
// 'bridging' variables in integral type.
template <typename Ty> using CmpxchgTy = __ESIMD_DNS::uint_type_t<sizeof(Ty)>;

template <typename Ty> inline Ty atomic_load(Ty *ptr) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  __ESIMD_UNSUPPORTED_ON_HOST;
  // TODO : Enable with unit test
  /* return sycl::bit_cast<Ty>(__atomic_load_n((CmpxchgTy<Ty> *)ptr,
                                               __ATOMIC_SEQ_CST)); */
#endif
}

template <typename Ty> inline Ty atomic_store(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  Ty ret = atomic_load<Ty>(ptr);
  __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
  return ret;
#endif
}

template <typename Ty> inline Ty atomic_add(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
  } else {
    // For Floating type
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = _old + val;
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n((CmpxchgTy<Ty> *)ptr, &_old_bits,
                                          _new_bits, false, __ATOMIC_SEQ_CST,
                                          __ATOMIC_SEQ_CST));
    return _old;
  }
#endif
}

template <typename Ty> inline Ty atomic_sub(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    return __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
  } else {
    // For Floating type
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = _old - val;
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n((CmpxchgTy<Ty> *)ptr, &_old_bits,
                                          _new_bits, false, __ATOMIC_SEQ_CST,
                                          __ATOMIC_SEQ_CST));
    return _old;
  }
#endif
}

template <typename Ty> inline Ty atomic_and(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral_v<Ty>);
  return __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> inline Ty atomic_or(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral_v<Ty>);
  return __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> inline Ty atomic_xor(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral_v<Ty>);
  return __atomic_fetch_xor(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> inline Ty atomic_min(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    Ty _old, _new;
    do {
      _old = *ptr;
      _new = std::min<Ty>(_old, val);
    } while (!__atomic_compare_exchange_n(ptr, &_old, _new, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _old;
  } else {
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = std::min(_old, val);
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n((CmpxchgTy<Ty> *)ptr, &_old_bits,
                                          _new_bits, false, __ATOMIC_SEQ_CST,
                                          __ATOMIC_SEQ_CST));
    return _old;
  }
#endif
}

template <typename Ty> inline Ty atomic_max(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    Ty _old, _new;
    do {
      _old = *ptr;
      _new = std::max<Ty>(_old, val);
    } while (!__atomic_compare_exchange_n(ptr, &_old, _new, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _old;
  } else {
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = std::max(_old, val);
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n((CmpxchgTy<Ty> *)(CmpxchgTy<Ty> *)ptr,
                                          &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _old;
  }
#endif
}

template <typename Ty>
inline Ty atomic_cmpxchg(Ty *ptr, Ty expected, Ty desired) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    Ty local = expected;
    __atomic_compare_exchange_n(ptr, &local, desired, false, __ATOMIC_SEQ_CST,
                                __ATOMIC_SEQ_CST);
    // if exchange occured, this means 'local=expected=*ptr'. So local
    // is returned as old val
    // if exchange did not occur, *ptr value compared against 'local'
    // is stored in 'local'. So local is returned as old val
    return local;
  } else {
    CmpxchgTy<Ty> desired_bits = *(CmpxchgTy<Ty> *)&desired;
    CmpxchgTy<Ty> local_bits = *(CmpxchgTy<Ty> *)&expected;
    __atomic_compare_exchange_n((CmpxchgTy<Ty> *)ptr, &local_bits, desired_bits,
                                false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return *((Ty *)&local_bits);
  }
#endif
}

inline void atomic_fence() {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

} // namespace __ESIMD_DNS

/// @endcond ESIMD_DETAIL
