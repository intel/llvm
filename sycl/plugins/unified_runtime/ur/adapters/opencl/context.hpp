//===--------- context.hpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"

namespace cl_adapter {
ur_result_t
getDevicesFromContext(ur_context_handle_t hContext,
                      std::unique_ptr<std::vector<cl_device_id>> &DevicesInCtx);
}
