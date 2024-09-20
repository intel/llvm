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

struct DeviceSanitizerReport;
struct AllocInfo;
struct StackTrace;
struct ValidateUSMResult;

void ReportBadFree(uptr Addr, const StackTrace &stack,
                   const std::shared_ptr<AllocInfo> &AllocInfo);

void ReportBadContext(uptr Addr, const StackTrace &stack,
                      const std::shared_ptr<AllocInfo> &AllocInfos);

void ReportDoubleFree(uptr Addr, const StackTrace &Stack,
                      const std::shared_ptr<AllocInfo> &AllocInfo);

void ReportMemoryLeak(const std::shared_ptr<AllocInfo> &AI);

// This type of error is usually unexpected mistake and doesn't have enough
// debug information
void ReportFatalError(const DeviceSanitizerReport &Report);

void ReportGenericError(const DeviceSanitizerReport &Report,
                        ur_kernel_handle_t Kernel);

void ReportUseAfterFree(const DeviceSanitizerReport &Report,
                        ur_kernel_handle_t Kernel, ur_context_handle_t Context);

void ReportInvalidKernelArgument(ur_kernel_handle_t Kernel, uint32_t ArgIndex,
                                 uptr Addr, const ValidateUSMResult &VR,
                                 StackTrace Stack);

} // namespace ur_sanitizer_layer
