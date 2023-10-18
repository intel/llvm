//===----------- common.hpp - Native CPU Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#pragma once

#include "ur/ur.hpp"

constexpr size_t MaxMessageSize = 256;

extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

#define DIE_NO_IMPLEMENTATION                                                  \
  if (PrintTrace) {                                                            \
    std::cerr << "Not Implemented : " << __FUNCTION__                          \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

#define CONTINUE_NO_IMPLEMENTATION                                             \
  if (PrintTrace) {                                                            \
    std::cerr << "Warning : Not Implemented : " << __FUNCTION__                \
              << " - File : " << __FILE__;                                     \
    std::cerr << " / Line : " << __LINE__ << std::endl;                        \
  }                                                                            \
  return UR_RESULT_SUCCESS;

#define CASE_UR_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    if (PrintTrace) {                                                          \
      std::cerr << std::endl                                                   \
                << "Unsupported UR case : " << #not_supported << " in "        \
                << __FUNCTION__ << ":" << __LINE__ << "(" << __FILE__ << ")"   \
                << std::endl;                                                  \
    }                                                                          \
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

/// ------ Error handling, matching OpenCL plugin semantics.
/// Taken from other adapter
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *pMessage);
} // namespace ur
} // namespace detail

// Base class to store common data
struct _ur_object {
  ur_shared_mutex Mutex;
};

struct RefCounted {
  std::atomic_uint32_t _refCount;
  void incrementReferenceCount() { _refCount++; }
  void decrementReferenceCount() { _refCount--; }
  RefCounted() : _refCount{1} {}
  uint32_t getReferenceCount() const { return _refCount; }
};
