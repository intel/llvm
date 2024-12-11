/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_report.hpp
 *
 */

#pragma once

#include "ur_api.h"

namespace ur_sanitizer_layer {

struct MsanErrorReport;

namespace msan {

void ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  ur_kernel_handle_t Kernel);

} // namespace msan
} // namespace ur_sanitizer_layer
