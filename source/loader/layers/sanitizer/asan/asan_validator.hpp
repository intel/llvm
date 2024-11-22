/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_validator.hpp
 *
 */
#pragma once

#include "asan_allocator.hpp"

namespace ur_sanitizer_layer {
namespace asan {

struct ValidateUSMResult {
    enum ErrorType {
        SUCCESS,
        NULL_POINTER,
        MAYBE_HOST_POINTER,
        RELEASED_POINTER,
        BAD_CONTEXT,
        BAD_DEVICE,
        OUT_OF_BOUNDS
    };
    ErrorType Type;
    std::shared_ptr<AllocInfo> AI;

    operator bool() { return Type != SUCCESS; }

    static ValidateUSMResult success() { return {SUCCESS, nullptr}; }

    static ValidateUSMResult fail(ErrorType Type,
                                  const std::shared_ptr<AllocInfo> &AI) {
        assert(Type != SUCCESS && "The error type shouldn't be SUCCESS");
        return {Type, AI};
    }

    static ValidateUSMResult fail(ErrorType Type) {
        assert(Type != SUCCESS && "The error type shouldn't be SUCCESS");
        return {Type, nullptr};
    }
};

ValidateUSMResult ValidateUSMPointer(ur_context_handle_t Context,
                                     ur_device_handle_t Device, uptr Ptr);

} // namespace asan
} // namespace ur_sanitizer_layer
