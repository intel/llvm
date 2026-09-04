//==---- abi_neutral.hpp - SYCL ABI-neutral return-type helpers -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_DEVICE_ONLY

#include <sycl/detail/string.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename T> struct ABINeutralT {
  using type = T;
};
// We need special handling of std::string to handle ABI incompatibility
// for get_info<>() when it returns std::string and vector<std::string>.
// For this purpose, get_info_impl<>() is created to handle special
// cases, and it is only called internally and not exposed to the user.
// The following ReturnType structure is intended for general return type,
// and special return types (std::string and vector of it).

template <> struct ABINeutralT<std::string> {
  using type = detail::string;
};

template <> struct ABINeutralT<std::vector<std::string>> {
  using type = std::vector<detail::string>;
};

template <typename T> using ABINeutralT_t = typename ABINeutralT<T>::type;

template <typename ParamT> auto convert_to_abi_neutral(ParamT &&Info) {
  using ParamDecayT = std::decay_t<ParamT>;
  if constexpr (std::is_same_v<ParamDecayT, std::string>) {
    return detail::string{Info};
  } else if constexpr (std::is_same_v<ParamDecayT, std::vector<std::string>>) {
    std::vector<detail::string> Res;
    Res.reserve(Info.size());
    for (std::string &Str : Info) {
      Res.push_back(detail::string{Str});
    }
    return Res;
  } else {
    return std::forward<ParamT>(Info);
  }
}

template <typename ParamT> auto convert_from_abi_neutral(ParamT &&Info) {
  using ParamNoRef = std::remove_reference_t<ParamT>;
  if constexpr (std::is_same_v<ParamNoRef, detail::string>) {
    return Info.c_str();
  } else if constexpr (std::is_same_v<ParamNoRef,
                                      std::vector<detail::string>>) {
    std::vector<std::string> Res;
    Res.reserve(Info.size());
    for (detail::string &Str : Info) {
      Res.push_back(Str.c_str());
    }
    return Res;
  } else {
    return std::forward<ParamT>(Info);
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif //__SYCL_DEVICE_ONLY
