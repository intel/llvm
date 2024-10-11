/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_statistics.hpp
 *
 */

#pragma once

#include "common.hpp"

namespace ur_sanitizer_layer {

struct AsanStats;

struct AsanStatsWrapper {

    AsanStatsWrapper();
    ~AsanStatsWrapper();

    void UpdateUSMMalloced(uptr MallocedSize, uptr RedzoneSize);
    void UpdateUSMFreed(uptr FreedSize);
    void UpdateUSMRealFreed(uptr FreedSize, uptr RedzoneSize);

    void UpdateShadowMmaped(uptr ShadowSize);
    void UpdateShadowMalloced(uptr ShadowSize);
    void UpdateShadowFreed(uptr ShadowSize);

    void Print(ur_context_handle_t Context);

  private:
    AsanStats *Stat;
};

} // namespace ur_sanitizer_layer
