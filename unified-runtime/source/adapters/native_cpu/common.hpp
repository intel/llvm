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

#include <chrono>

#include "common/ur_ref_counter.hpp"
#include "logger/ur_logger.hpp"
#include "ur/ur.hpp"

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
