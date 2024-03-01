/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_report.hpp
 *
 */

#pragma once

#include "common.hpp"

#include <memory>

namespace ur_sanitizer_layer {

class DeviceSanitizerReport;
class AllocInfo;
class StackTrace;

void ReportBadFree(uptr Addr, const StackTrace &stack,
                   std::shared_ptr<AllocInfo> AllocInfo);

void ReportBadFreeContext(
    uptr Addr, const StackTrace &stack,
    const std::vector<std::shared_ptr<AllocInfo>> &AllocInfos);

void ReportDoubleFree(uptr Addr, const StackTrace &Stack,
                      std::shared_ptr<AllocInfo> AllocInfo);

void ReportGenericError(const DeviceSanitizerReport &Report,
                        ur_kernel_handle_t Kernel, ur_context_handle_t Context,
                        ur_device_handle_t Device);

} // namespace ur_sanitizer_layer
