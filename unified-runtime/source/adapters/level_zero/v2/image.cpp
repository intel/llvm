//===--------- image.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include "../image_common.hpp"
#include "../ur_interface_loader.hpp"
#include "../v2/context.hpp"
#include "../v2/memory.hpp"
#include "logger/ur_logger.hpp"
#include "queue_api.hpp"
#include "queue_handle.hpp"

namespace ur::level_zero {

ur_result_t
urBindlessImagesImageFreeExp([[maybe_unused]] ur_context_handle_t hContext,
                             [[maybe_unused]] ur_device_handle_t hDevice,
                             ur_exp_image_mem_native_handle_t hImageMem) {
  ur_bindless_mem_handle_t *urImg =
      reinterpret_cast<ur_bindless_mem_handle_t *>(hImageMem);
  delete urImg;
  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
