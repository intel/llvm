//==---------- device_sanitizer_report.hpp - Device Sanitizer -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cinttypes>

namespace ur_san_layer {

enum class DeviceSanitizerErrorType : int32_t {
    UNKNOWN,
    OUT_OF_BOUND,
    MISALIGNED,
    USE_AFTER_FREE,
    OUT_OF_SHADOW_BOUND,
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

// NOTE Layout of this structure should be aligned with the one in
// sycl/include/sycl/detail/device_sanitizer_report.hpp
struct DeviceSanitizerReport {
    int Flag = 0;

    char File[256 + 1] = "";
    char Func[128 + 1] = "";

    int32_t Line = 0;

    uint64_t GID0 = 0;
    uint64_t GID1 = 0;
    uint64_t GID2 = 0;

    uint64_t LID0 = 0;
    uint64_t LID1 = 0;
    uint64_t LID2 = 0;

    bool IsWrite = false;
    uint32_t AccessSize = 0;
    DeviceSanitizerMemoryType MemoryType;
    DeviceSanitizerErrorType ErrorType;

    bool IsRecover = false;
};

const char *DeviceSanitizerFormat(DeviceSanitizerMemoryType MemoryType) {
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

const char *DeviceSanitizerFormat(DeviceSanitizerErrorType ErrorType) {
    switch (ErrorType) {
    case DeviceSanitizerErrorType::OUT_OF_BOUND:
        return "out-of-bound-access";
    case DeviceSanitizerErrorType::MISALIGNED:
        return "misaligned-access";
    case DeviceSanitizerErrorType::USE_AFTER_FREE:
        return "use-after-free";
    case DeviceSanitizerErrorType::OUT_OF_SHADOW_BOUND:
        return "out-of-shadow-bound-access";
    default:
        return "unknown-error";
    }
}

} // namespace ur_san_layer
