// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

#include <level_zero/layers/zel_tracing_api.h>
#include <loader/ze_loader.h>

#include <memory>

std::shared_ptr<_zel_tracer_handle_t>
enableTracing(zel_core_callbacks_t prologueCallbacks,
              zel_core_callbacks_t epilogueCallbacks) {
    EXPECT_EQ(zeInit(ZE_INIT_FLAG_GPU_ONLY), ZE_RESULT_SUCCESS);

    zel_tracer_desc_t tracer_desc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC,
                                     nullptr, nullptr};
    zel_tracer_handle_t tracer = nullptr;
    EXPECT_EQ(zelTracerCreate(&tracer_desc, &tracer), ZE_RESULT_SUCCESS);

    EXPECT_EQ(zelTracerSetPrologues(tracer, &prologueCallbacks),
              ZE_RESULT_SUCCESS);
    EXPECT_EQ(zelTracerSetEpilogues(tracer, &epilogueCallbacks),
              ZE_RESULT_SUCCESS);
    EXPECT_EQ(zelTracerSetEnabled(tracer, true), ZE_RESULT_SUCCESS);

    return std::shared_ptr<_zel_tracer_handle_t>(
        tracer, [](zel_tracer_handle_t tracer) { zelTracerDestroy(tracer); });
}
