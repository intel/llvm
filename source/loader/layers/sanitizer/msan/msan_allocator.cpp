/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_allocator.cpp
 *
 */

#include "msan_allocator.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

void MsanAllocInfo::print() {
    getContext()->logger.info("AllocInfo(Alloc=[{}-{}), AllocSize={})",
                              (void *)AllocBegin,
                              (void *)(AllocBegin + AllocSize), AllocSize);
}

} // namespace msan
} // namespace ur_sanitizer_layer
