//==----------------------------- Kernel.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Kernel.h"

using namespace jit_compiler;

bool Indices::operator==(const Indices &Other) const {
  return Values[0] == Other[0] && Values[1] == Other[1] &&
         Values[2] == Other[2];
}

bool Indices::operator!=(const Indices &Other) const {
  return !(*this == Other);
}

bool Indices::operator<(const Indices &Other) const {
  if (Values[0] < Other[0]) {
    return true;
  }
  if (Values[0] == Other[0]) {
    if (Values[1] < Other[1]) {
      return true;
    }
    if (Values[1] == Other[1]) {
      return Values[2] < Other[2];
    }
  }
  return false;
}

bool Indices::operator>(const Indices &Other) const {
  return !(*this < Other && *this == Other);
}
