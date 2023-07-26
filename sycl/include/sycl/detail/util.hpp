//===-- util.hpp - Shared SYCL runtime utilities interface -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_DEVICE_ONLY

#include <sycl/detail/defines.hpp>

#include <cstring>
#include <mutex>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Groups and provides access to all the locks used the SYCL runtime.
class Sync {
public:
  /// Retuns a reference to the global lock. The global lock is the default
  /// SYCL runtime lock guarding use of most SYCL runtime resources.
  static std::mutex &getGlobalLock() { return getInstance().GlobalLock; }

private:
  static Sync &getInstance();
  std::mutex GlobalLock;
};

// TempAssignGuard is the class for a guard object that will assign some OTHER
// variable to a temporary value but restore it when the guard itself goes out
// of scope.
template <typename T> struct TempAssignGuard {
  T &field;
  T restoreValue;
  TempAssignGuard(T &fld, T tempVal) : field(fld), restoreValue(fld) {
    field = tempVal;
  }
  ~TempAssignGuard() { field = restoreValue; }
};

// const char* key hash for STL maps
struct HashCStr {
  size_t operator()(const char *S) const {
    constexpr size_t Prime = 31;
    size_t Res = 0;
    char Ch = 0;

    for (; (Ch = *S); S++) {
      Res += Ch + (Prime * Res);
    }
    return Res;
  }
};

// const char* key comparison for STL maps
struct CmpCStr {
  bool operator()(const char *A, const char *B) const {
    return std::strcmp(A, B) == 0;
  }
};

using SerializedObj = std::vector<unsigned char>;

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif //__SYCL_DEVICE_ONLY
