/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_libdevice.hpp
 *
 */

#pragma once

#include <cstdint>

#if !defined(__SPIR__) && !defined(__SPIRV__)
namespace ur_sanitizer_layer {
#endif // !__SPIR__ && !__SPIRV__

enum class DeviceType : uint32_t { UNKNOWN = 0, CPU, GPU_PVC, GPU_DG2 };

inline const char *ToString(DeviceType Type) {
    switch (Type) {
    case DeviceType::UNKNOWN:
        return "UNKNOWN";
    case DeviceType::CPU:
        return "CPU";
    case DeviceType::GPU_PVC:
        return "PVC";
    case DeviceType::GPU_DG2:
        return "DG2";
    default:
        return "UNKNOWN";
    }
}

enum class ErrorType : int32_t {
    UNKNOWN,
    OUT_OF_BOUNDS,
    MISALIGNED,
    USE_AFTER_FREE,
    OUT_OF_SHADOW_BOUNDS,
    UNKNOWN_DEVICE,
    NULL_POINTER,
};

inline const char *ToString(ErrorType ErrorType) {
    switch (ErrorType) {
    case ErrorType::OUT_OF_BOUNDS:
        return "out-of-bounds-access";
    case ErrorType::MISALIGNED:
        return "misaligned-access";
    case ErrorType::USE_AFTER_FREE:
        return "use-after-free";
    case ErrorType::OUT_OF_SHADOW_BOUNDS:
        return "out-of-shadow-bounds-access";
    case ErrorType::UNKNOWN_DEVICE:
        return "unknown-device";
    case ErrorType::NULL_POINTER:
        return "null-pointer-access";
    default:
        return "unknown-error";
    }
}

enum class MemoryType : int32_t {
    UNKNOWN,
    USM_DEVICE,
    USM_HOST,
    USM_SHARED,
    LOCAL,
    PRIVATE,
    MEM_BUFFER,
    DEVICE_GLOBAL,
};

inline const char *ToString(MemoryType MemoryType) {
    switch (MemoryType) {
    case MemoryType::USM_DEVICE:
        return "Device USM";
    case MemoryType::USM_HOST:
        return "Host USM";
    case MemoryType::USM_SHARED:
        return "Shared USM";
    case MemoryType::LOCAL:
        return "Local Memory";
    case MemoryType::PRIVATE:
        return "Private Memory";
    case MemoryType::MEM_BUFFER:
        return "Memory Buffer";
    case MemoryType::DEVICE_GLOBAL:
        return "Device Global";
    default:
        return "Unknown Memory";
    }
}

#if !defined(__SPIR__) && !defined(__SPIRV__)
} // namespace ur_sanitizer_layer
#endif // !__SPIR__ && !__SPIRV__
