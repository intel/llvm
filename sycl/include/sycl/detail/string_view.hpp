//==-------------- string_view.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/string.hpp>

#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail {

// This class and detail::string class are intended to support
// different ABIs between libsycl and the user program.
// This class is not intended to replace std::string_view for general purpose
// usage.

class string_view {
  const char *str = nullptr;
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  size_t len = 0;
#endif

public:
  string_view() noexcept = default;
  string_view(const string_view &strn) noexcept = default;
  string_view(string_view &&strn) noexcept = default;
  string_view(std::string_view strn) noexcept
      : str(strn.data())
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
        ,
        len(strn.size())
#endif
  {
  }
  string_view(const sycl::detail::string &strn) noexcept
      : str(strn.c_str())
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
        ,
        len(strlen(strn.c_str()))
#endif
  {
  }

  string_view &operator=(string_view &&strn) noexcept = default;
  string_view &operator=(const string_view &strn) noexcept = default;

  string_view &operator=(std::string_view strn) noexcept {
    str = strn.data();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    len = strn.size();
#endif
    return *this;
  }

  string_view &operator=(const sycl::detail::string &strn) noexcept {
    str = strn.c_str();
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    len = strlen(strn.c_str());
#endif
    return *this;
  }

  const char *data() const noexcept { return str ? str : ""; }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  std::string_view toStringView() const noexcept {
    return std::string_view(str, len);
  }
#endif

  friend bool operator==(string_view lhs, std::string_view rhs) noexcept {
    return rhs == lhs.data();
  }
  friend bool operator==(std::string_view lhs, string_view rhs) noexcept {
    return lhs == rhs.data();
  }

  friend bool operator!=(string_view lhs, std::string_view rhs) noexcept {
    return rhs != lhs.data();
  }
  friend bool operator!=(std::string_view lhs, string_view rhs) noexcept {
    return lhs != rhs.data();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
