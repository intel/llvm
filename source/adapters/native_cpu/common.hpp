//===----------- common.hpp - Native CPU Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "logger/ur_logger.hpp"
#include "ur/ur.hpp"
#include <chrono>

constexpr size_t MaxMessageSize = 256;

extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

#define DIE_NO_IMPLEMENTATION                                                  \
  do {                                                                         \
    logger::error("Not Implemented : {} - File : {} / Line : {}",              \
                  __FUNCTION__, __FILE__, __LINE__);                           \
                                                                               \
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;                                \
  } while (false);

#define CONTINUE_NO_IMPLEMENTATION                                             \
  do {                                                                         \
    logger::warning("Not Implemented : {} - File : {} / Line : {}",            \
                    __FUNCTION__, __FILE__, __LINE__);                         \
    return UR_RESULT_SUCCESS;                                                  \
  } while (false);

#define CASE_UR_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    logger::error("Unsupported UR case : {} in {}:{}({})", #not_supported,     \
                  __FUNCTION__, __LINE__, __FILE__);                           \
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

// Todo: replace this with a common helper once it is available
struct RefCounted {
  std::atomic_uint32_t _refCount;
  uint32_t incrementReferenceCount() { return ++_refCount; }
  uint32_t decrementReferenceCount() { return --_refCount; }
  RefCounted() : _refCount{1} {}
  uint32_t getReferenceCount() const { return _refCount; }
};

template <typename T> inline void decrementOrDelete(T *refC) {
  if (refC->decrementReferenceCount() == 0)
    delete refC;
}

inline uint64_t get_timestamp() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

namespace native_cpu {

inline void *aligned_malloc(size_t alignment, size_t size) {
  void *ptr = nullptr;
#ifdef _MSC_VER
  ptr = _aligned_malloc(size, alignment);
#else
  ptr = std::aligned_alloc(alignment, size);
#endif
  return ptr;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

} // namespace native_cpu
