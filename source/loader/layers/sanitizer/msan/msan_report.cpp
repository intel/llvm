/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_report.cpp
 *
 */

#include "msan_report.hpp"
#include "msan_libdevice.hpp"

#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

void ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  ur_kernel_handle_t Kernel) {
    const char *File = Report.File[0] ? Report.File : "<unknown file>";
    const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
    auto KernelName = GetKernelName(Kernel);

    // Try to demangle the kernel name
    KernelName = DemangleName(KernelName);

    getContext()->logger.always(
        "====WARNING: DeviceSanitizer: use-of-uninitialized-value");
    getContext()->logger.always(
        "use of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
        "{}, {})",
        Report.AccessSize, KernelName.c_str(), Report.LID0, Report.LID1,
        Report.LID2, Report.GID0, Report.GID1, Report.GID2);
    getContext()->logger.always("  #0 {} {}:{}", Func, File, Report.Line);
}

} // namespace msan
} // namespace ur_sanitizer_layer
