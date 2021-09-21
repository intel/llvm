//===-- util.hpp - Shared SYCL runtime utilities interface -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_DEVICE_ONLY

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/stl.hpp>

#include <cstring>
#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#endif //__SYCL_DEVICE_ONLY
