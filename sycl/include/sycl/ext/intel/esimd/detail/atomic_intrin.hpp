//==-------- atomic_intrin.hpp - Atomic intrinsic definition file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/exception.hpp>

namespace __ESIMD_DNS {

// This function implements atomic update of pre-existing variable in the
// absense of C++ 20's atomic_ref.

// __atomic_* functions support only integral types. In order to
// support floating types for certain operations like min/max,
// 'cmpxchg' operation is applied for result values using
// 'bridging' variables in integral type.
template <typename Ty> using CmpxchgTy = __ESIMD_DNS::uint_type_t<sizeof(Ty)>;

template <typename Ty> Ty atomic_load(Ty *ptr) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  return __atomic_load(ptr, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> Ty atomic_store(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  __atomic_store(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> Ty atomic_add_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    return __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
  } else {
    // For Floating type
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = _old + val;
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n(ptr, &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _new;
  }
#endif
}

template <typename Ty> Ty atomic_sub_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    return __atomic_sub_fetch(ptr, val, __ATOMIC_SEQ_CST);
  } else {
    // For Floating type
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = _old - val;
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n(ptr, &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _new;
  }
#endif
}

template <typename Ty> Ty atomic_and_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral<Ty>::value);
  return __atomic_and_fetch(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> Ty atomic_or_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral<Ty>::value);
  return __atomic_or_fetch(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> Ty atomic_xor_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  static_assert(std::is_integral<Ty>::value);
  return __atomic_xor_fetch(ptr, val, __ATOMIC_SEQ_CST);
#endif
}

template <typename Ty> Ty atomic_min(Ty *ptr, Ty val) {
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
    return _new;
  } else {
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = std::min(_old, val);
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n(ptr, &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _new;
  }
#endif
}

template <typename Ty> Ty atomic_max(Ty *ptr, Ty val) {
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
    return _new;
  } else {
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = *ptr;
      _new = std::max(_old, val);
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n(ptr, &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _new;
  }
#endif
}

template <typename Ty> Ty atomic_cmpxchg(Ty *ptr, Ty desired, Ty expected) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  if constexpr (std::is_integral_v<Ty>) {
    Ty _old = expected;
    __atomic_compare_exchange_n(ptr, &_old, desired, false, __ATOMIC_SEQ_CST,
                                __ATOMIC_SEQ_CST);
    return *ptr;
  } else {
    Ty _old, _new;
    CmpxchgTy<Ty> _old_bits, _new_bits;
    do {
      _old = expected;
      _new = desired;
      _old_bits = *(CmpxchgTy<Ty> *)&_old;
      _new_bits = *(CmpxchgTy<Ty> *)&_new;
    } while (!__atomic_compare_exchange_n(ptr, &_old_bits, _new_bits, false,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return _new;
  }
#endif
}

} // namespace __ESIMD_DNS

/// @endcond ESIMD_DETAIL
