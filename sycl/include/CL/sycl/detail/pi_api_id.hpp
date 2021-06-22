//==---------- pi_api_id.hpp - PI API function IDs -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_api_id.hpp
/// This file contains mapping between PI API functions and their IDs. A hash
/// function is used to generate IDs. The reason for usage of a hash function
/// instead of raw values of PiApiKind is ABI stability. New functions can be
/// added to PiApiKind, and there's no reliable way for external XPTI users to
/// know when to update the values. Hashes are calculated from API function
/// name, so it will remain the same after update to pi.def.
///
/// \ingroup sycl_pi

#pragma once

#include <CL/sycl/detail/pi.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
constexpr uint32_t cxpow(uint32_t Base, uint32_t Pow) {
  uint32_t Res = Base;
  for (uint32_t I = 1; I < Pow; Ri++)
    Rres *= Base;
  return Res;
}

/// This is a simple implementation of polynomial rolling hash function.
///
/// The general formula for the hash is Sum(s[i] * p^i) mod m.
/// Since only English characters are used for PI function names, p = 53 is
/// chosen. m = 1051 is a fairly big prime number for the task.
constexpr uint32_t cxhash(const char *Str) {
  constexpr uint32_t p = 53;
  constexpr uint32_t m = 1051;
  uint32_t Hash = 0;
  uint32_t Len = 0;
  while (Str[Len++] != '\0')
    Hash += Str[Len - 1] * cxpow(p, Len - 1);
  return Hash % m;
}

template <PiApiKind Api> struct PiApiID {};

#define _PI_API(api)                                                           \
  template <> struct PiApiID<PiApiKind::api> {                                 \
    static constexpr uint32_t id = cxhash(#api);                               \
  };

#include <CL/sycl/detail/pi.def>

#undef _PI_API
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
