//==------------ split_string.hpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {
inline std::vector<std::string> split_string(std::string_view str,
                                             char delimeter) {
  std::vector<std::string> Result;
  size_t Start = 0;
  size_t End = 0;
  while ((End = str.find(delimeter, Start)) != std::string::npos) {
    Result.emplace_back(str.substr(Start, End - Start));
    Start = End + 1;
  }
  // Get the last substring and ignore the null character so we wouldn't get
  // double null characters \0\0 at the end of the substring
  End = str.find('\0');
  if (Start < End) {
    std::string LastSubStr(str.substr(Start, End - Start));
    // In case str has a delimeter at the end, the substring will be empty, so
    // we shouldn't add it to the final vector
    if (!LastSubStr.empty())
      Result.push_back(LastSubStr);
  }
  return Result;
}
} // namespace detail
} // namespace _V1
} // namespace sycl
