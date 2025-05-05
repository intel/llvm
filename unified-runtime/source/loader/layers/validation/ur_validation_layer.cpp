/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_validation_layer.cpp
 *
 */
#include "ur_validation_layer.hpp"
#include "ur_leak_check.hpp"

#include <cassert>

namespace ur_validation_layer {
context_t *getContext() { return context_t::get_direct(); }

///////////////////////////////////////////////////////////////////////////////
context_t::context_t()
    : logger(logger::create_logger("validation")),
      refCountContext(new RefCountContext()) {}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

// Some adapters don't support all the queries yet, we should be lenient and
// just not attempt to validate in those cases to preserve functionality.
#define RETURN_ON_FAILURE(result)                                              \
  if (result == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION ||                     \
      result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)                           \
    return UR_RESULT_SUCCESS;                                                  \
  if (result != UR_RESULT_SUCCESS) {                                           \
    getContext()->logger.error("Unexpected non-success result code from {}",   \
                               #result);                                       \
    assert(0);                                                                 \
    return result;                                                             \
  }

ur_result_t bounds(ur_mem_handle_t buffer, size_t offset, size_t size) {
  auto pfnMemGetInfo = getContext()->urDdiTable.Mem.pfnGetInfo;

  size_t bufferSize = 0;
  RETURN_ON_FAILURE(pfnMemGetInfo(buffer, UR_MEM_INFO_SIZE, sizeof(bufferSize),
                                  &bufferSize, nullptr));

  if (size + offset > bufferSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t bounds(ur_mem_handle_t buffer, ur_rect_offset_t offset,
                   ur_rect_region_t region) {
  auto pfnMemGetInfo = getContext()->urDdiTable.Mem.pfnGetInfo;

  size_t bufferSize = 0;
  RETURN_ON_FAILURE(pfnMemGetInfo(buffer, UR_MEM_INFO_SIZE, sizeof(bufferSize),
                                  &bufferSize, nullptr));

  if (offset.x >= bufferSize || offset.y >= bufferSize ||
      offset.z >= bufferSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  if ((region.width + offset.x) * (region.height + offset.y) *
          (region.depth + offset.z) >
      bufferSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t bounds(ur_queue_handle_t queue, const void *ptr, size_t offset,
                   size_t size) {
  auto pfnQueueGetInfo = getContext()->urDdiTable.Queue.pfnGetInfo;
  auto pfnUSMGetMemAllocInfo = getContext()->urDdiTable.USM.pfnGetMemAllocInfo;

  ur_context_handle_t urContext = nullptr;
  RETURN_ON_FAILURE(pfnQueueGetInfo(queue, UR_QUEUE_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t), &urContext,
                                    nullptr));
  ur_usm_type_t usmType = UR_USM_TYPE_UNKNOWN;
  RETURN_ON_FAILURE(pfnUSMGetMemAllocInfo(urContext, ptr,
                                          UR_USM_ALLOC_INFO_TYPE,
                                          sizeof(usmType), &usmType, nullptr));

  // We can't reliably get size info about pointers that didn't come from the
  // USM alloc entry points.
  if (usmType == UR_USM_TYPE_UNKNOWN) {
    return UR_RESULT_SUCCESS;
  }

  size_t allocSize = 0;
  RETURN_ON_FAILURE(
      pfnUSMGetMemAllocInfo(urContext, ptr, UR_USM_ALLOC_INFO_SIZE,
                            sizeof(allocSize), &allocSize, nullptr));

  if (size + offset > allocSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t boundsImage(ur_mem_handle_t image, ur_rect_offset_t origin,
                        ur_rect_region_t region) {
  auto pfnMemImageGetInfo = getContext()->urDdiTable.Mem.pfnImageGetInfo;

  size_t width = 0;
  RETURN_ON_FAILURE(pfnMemImageGetInfo(image, UR_IMAGE_INFO_WIDTH,
                                       sizeof(width), &width, nullptr));
  if (region.width + origin.x > width) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  size_t height = 0;
  RETURN_ON_FAILURE(pfnMemImageGetInfo(image, UR_IMAGE_INFO_HEIGHT,
                                       sizeof(height), &height, nullptr));

  // Some adapters return a height and depth of 0 for images that don't have
  // those dimensions, but regions for enqueue operations must set these to
  // 1, so we need to make this adjustment to properly validate.
  if (height == 0) {
    height = 1;
  }

  if (region.height + origin.y > height) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  size_t depth = 0;
  RETURN_ON_FAILURE(pfnMemImageGetInfo(image, UR_IMAGE_INFO_DEPTH,
                                       sizeof(depth), &depth, nullptr));
  if (depth == 0) {
    depth = 1;
  }

  if (region.depth + origin.z > depth) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  return UR_RESULT_SUCCESS;
}

#undef RETURN_ON_FAILURE

} // namespace ur_validation_layer
