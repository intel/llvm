/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_validator.cpp
 *
 */

#include "asan_validator.hpp"
#include "asan_interceptor.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"

namespace ur_sanitizer_layer {
namespace asan {

namespace {

bool IsSameDevice(ur_device_handle_t Device1, ur_device_handle_t Device2) {
    if (Device1 == Device2) {
        return true;
    }
    auto RootDevice1 = GetParentDevice(Device1);
    RootDevice1 = RootDevice1 ? RootDevice1 : Device1;
    auto RootDevice2 = GetParentDevice(Device2);
    RootDevice2 = RootDevice2 ? RootDevice2 : Device2;
    if (RootDevice1 == RootDevice2) {
        return true;
    }
    return false;
}

} // namespace

ValidateUSMResult ValidateUSMPointer(ur_context_handle_t Context,
                                     ur_device_handle_t Device, uptr Ptr) {
    assert(Ptr != 0 && "Don't validate nullptr here");

    auto AllocInfoItOp = getAsanInterceptor()->findAllocInfoByAddress(Ptr);
    if (!AllocInfoItOp) {
        auto DI = getAsanInterceptor()->getDeviceInfo(Device);
        bool IsSupportSharedSystemUSM = DI->IsSupportSharedSystemUSM;
        if (IsSupportSharedSystemUSM) {
            // maybe it's host pointer
            return ValidateUSMResult::success();
        }
        return ValidateUSMResult::fail(ValidateUSMResult::MAYBE_HOST_POINTER);
    }

    auto AllocInfo = AllocInfoItOp.value()->second;

    if (AllocInfo->Context != Context) {
        return ValidateUSMResult::fail(ValidateUSMResult::BAD_CONTEXT,
                                       AllocInfo);
    }

    if (AllocInfo->Device && !IsSameDevice(AllocInfo->Device, Device)) {
        return ValidateUSMResult::fail(ValidateUSMResult::BAD_DEVICE,
                                       AllocInfo);
    }

    if (AllocInfo->IsReleased) {
        return ValidateUSMResult::fail(ValidateUSMResult::RELEASED_POINTER,
                                       AllocInfo);
    }

    if (Ptr < AllocInfo->UserBegin || Ptr >= AllocInfo->UserEnd) {
        return ValidateUSMResult::fail(ValidateUSMResult::OUT_OF_BOUNDS,
                                       AllocInfo);
    }

    return ValidateUSMResult::success();
}

} // namespace asan
} // namespace ur_sanitizer_layer
