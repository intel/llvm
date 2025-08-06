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

extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

#define DIE_NO_IMPLEMENTATION                                                  \
  do {                                                                         \
    UR_LOG(ERR, "Not Implemented : {}", __FUNCTION__)                          \
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;                                \
  } while (false)

#define CONTINUE_NO_IMPLEMENTATION                                             \
  do {                                                                         \
    UR_LOG(WARN, "Not Implemented : {}", __FUNCTION__)                         \
    return UR_RESULT_SUCCESS;                                                  \
  } while (false)

#define CASE_UR_UNSUPPORTED(not_supported)                                     \
  case not_supported:                                                          \
    UR_LOG(ERR, "Unsupported UR case : {} in {}", #not_supported,              \
           __FUNCTION__)                                                       \
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

namespace ur::native_cpu {
struct ddi_getter {
  static const ur_dditable_t *value();
};
using handle_base = ur::handle_base<ddi_getter>;
} // namespace ur::native_cpu

// Base class to store common data
struct ur_object {
  ur_shared_mutex Mutex;
};

template <typename T> inline void decrementOrDelete(T *refC) {
  if (refC->RefCount.release())
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

// In many cases we require aligned memory without being told what the alignment
// requirement is. This helper function returns maximally aligned memory based
// on the size.
inline void *aligned_malloc(size_t size) {
  constexpr size_t max_alignment = 16 * sizeof(double);
  size_t alignment = max_alignment;
  while (alignment > size) {
    alignment >>= 1;
  }
  // aligned_malloc requires size to be a multiple of alignment; round up.
  size = (size + alignment - 1) & ~(alignment - 1);
  return aligned_malloc(alignment, size);
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

} // namespace native_cpu
