/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file device_sanitizer_report.hpp
 *
 */

#pragma once

#include <cinttypes>

namespace ur_sanitizer_layer {

enum class DeviceSanitizerErrorType : int32_t {
    UNKNOWN,
    OUT_OF_BOUNDS,
    MISALIGNED,
    USE_AFTER_FREE,
    OUT_OF_SHADOW_BOUNDS,
};

enum class DeviceSanitizerMemoryType : int32_t {
    UNKNOWN,
    USM_DEVICE,
    USM_HOST,
    USM_SHARED,
    LOCAL,
    PRIVATE,
    MEM_BUFFER,
};

struct DeviceSanitizerReport {
    int Flag = 0;

    char File[256 + 1] = "";
    char Func[256 + 1] = "";

    int32_t Line = 0;

    uint64_t GID0 = 0;
    uint64_t GID1 = 0;
    uint64_t GID2 = 0;

    uint64_t LID0 = 0;
    uint64_t LID1 = 0;
    uint64_t LID2 = 0;

    uint64_t Addr = 0;
    bool IsWrite = false;
    uint32_t AccessSize = 0;
    DeviceSanitizerMemoryType MemoryType;
    DeviceSanitizerErrorType ErrorType;

    bool IsRecover = false;
};

inline const char *DeviceSanitizerFormat(DeviceSanitizerMemoryType MemoryType) {
    switch (MemoryType) {
    case DeviceSanitizerMemoryType::USM_DEVICE:
        return "USM Device Memory";
    case DeviceSanitizerMemoryType::USM_HOST:
        return "USM Host Memory";
    case DeviceSanitizerMemoryType::USM_SHARED:
        return "USM Shared Memory";
    case DeviceSanitizerMemoryType::LOCAL:
        return "Local Memory";
    case DeviceSanitizerMemoryType::PRIVATE:
        return "Private Memory";
    case DeviceSanitizerMemoryType::MEM_BUFFER:
        return "Memory Buffer";
    default:
        return "Unknown Memory";
    }
}

inline const char *DeviceSanitizerFormat(DeviceSanitizerErrorType ErrorType) {
    switch (ErrorType) {
    case DeviceSanitizerErrorType::OUT_OF_BOUNDS:
        return "out-of-bounds-access";
    case DeviceSanitizerErrorType::MISALIGNED:
        return "misaligned-access";
    case DeviceSanitizerErrorType::USE_AFTER_FREE:
        return "use-after-free";
    case DeviceSanitizerErrorType::OUT_OF_SHADOW_BOUNDS:
        return "out-of-shadow-bounds-access";
    default:
        return "unknown-error";
    }
}

} // namespace ur_sanitizer_layer
