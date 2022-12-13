//==------------------- memops2d_utils.hpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

using namespace sycl;

enum OperationPath {
  Expanded,
  ExpandedDependsOn,
  ShortcutNoEvent,
  ShortcutOneEvent,
  ShortcutEventList
};

std::string operationPathToString(OperationPath PathKind) {
  switch (PathKind) {
  case Expanded:
    return "no shortcut and no depends_on";
  case ExpandedDependsOn:
    return "no shortcut";
  case ShortcutNoEvent:
    return "shortcut with no dependency events";
  case ShortcutOneEvent:
    return "shortcut with one dependency event";
  case ShortcutEventList:
    return "shortcut with dependency event list";
  default:
    return "UNKNOWN";
  }
}

std::string usmAllocTypeToString(usm::alloc AllocKind) {
  switch (AllocKind) {
  case usm::alloc::device:
    return "device USM allocation";
  case usm::alloc::host:
    return "host USM allocation";
  case usm::alloc::shared:
    return "shared USM allocation";
  default:
    return "UNKNOWN";
  }
}

struct TestStruct {
  int a;
  char b;

  bool operator==(const TestStruct &RHS) const {
    return a == RHS.a && b == RHS.b;
  }
  bool operator!=(const TestStruct &RHS) const { return !(*this == RHS); }
};

std::ostream &operator<<(std::ostream &Out, const TestStruct &RHS) {
  Out << '{' << RHS.a << ',' << RHS.b << '}';
  return Out;
}

template <usm::alloc AllocKind, OperationPath PathKind, typename T>
bool checkResult(T &Result, T &Expected, size_t Index,
                 std::string_view TestName) {
  if (Result != Expected) {
    std::cout << TestName << " (" << usmAllocTypeToString(AllocKind) << ", "
              << operationPathToString(PathKind) << ")\nValue at " << Index
              << " did not match the expected value; " << Result
              << " != " << Expected << std::endl;
    return false;
  }
  return true;
}
