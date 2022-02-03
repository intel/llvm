//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include <random>

inline static std::string getRandomString() {
  std::random_device Dev;
  std::mt19937 Range(Dev());
  std::uniform_int_distribution<std::mt19937::result_type> Dist(1, 255);

  size_t Size = Dist(Range);
  std::string Result = "";
  Result.resize(Size);

  for (char &C : Result) {
    C = static_cast<char>(Dist(Range));
  }

  return Result;
}
